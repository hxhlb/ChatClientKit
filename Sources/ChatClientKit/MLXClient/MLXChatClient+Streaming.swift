//
//  MLXChatClient+Streaming.swift
//  ChatClientKit
//
//  Created by GPT-5 Codex on 2025/11/10.
//

import CoreImage
import Foundation
@preconcurrency import MLXLLM
@preconcurrency import MLXLMCommon
@preconcurrency import MLXVLM

private struct UnsafeSendableBox<Value>: @unchecked Sendable {
    let value: Value

    init(_ value: Value) {
        self.value = value
    }
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
extension MLXChatClient {
    func streamingChatCompletionRequestExecute(
        body: ChatRequestBody,
        token: UUID
    ) async throws -> AnyAsyncSequence<ChatResponseChunk> {
        var userInput = userInput(body: body)
        let generateParameters = generateParameters(body: body)
        let container = try await loadContainer(adjusting: &userInput)
        let toolSpecs = userInput.tools
        return try await container.perform(nonSendable: userInput) { context, lockedInput in
            let input = try await context.processor.prepare(input: lockedInput)

            let toolCallFormat = context.configuration.toolCallFormat ?? .json
            let streamInput = UnsafeSendableBox(input)
            let streamContext = UnsafeSendableBox(context)
            let streamParameters = generateParameters
            let streamToolSpecs = toolSpecs

            return AsyncThrowingStream { continuation in
                let workerTask = Task(priority: .userInitiated) {
                    defer { MLXChatClientQueue.shared.release(token: token) }

                    let input = streamInput.value
                    let context = streamContext.value

                    let toolProcessor: ToolCallProcessor? = if let streamToolSpecs, !streamToolSpecs.isEmpty {
                        ToolCallProcessor(format: toolCallFormat, tools: streamToolSpecs)
                    } else {
                        nil
                    }

                    var latestOutputLength = 0
                    var shouldRemoveLeadingWhitespace = true
                    var decoder = ChunkDecoder(context: context, input: input)

                    do {
                        let (tokenStream, generationTask) = try MLXLMCommon.generateTokensTask(
                            input: input,
                            parameters: streamParameters,
                            context: context
                        )
                        defer { generationTask.cancel() }

                        var generatedTokens: [Int] = []
                        generationLoop: for await generation in tokenStream {
                            if Task.isCancelled {
                                logger.debug("cancelling current inference due to Task.isCancelled")
                                generationTask.cancel()
                                break generationLoop
                            }

                            guard case let .token(token) = generation else {
                                continue
                            }

                            generatedTokens.append(token)
                            let decodeResult = decoder.decode(
                                tokens: generatedTokens,
                                latestOutputLength: &latestOutputLength,
                                shouldRemoveLeadingWhitespace: &shouldRemoveLeadingWhitespace
                            )

                            if let generatedChunk = decodeResult.chunk {
                                for choice in generatedChunk.choices {
                                    if let reasoning = choice.delta.reasoningContent {
                                        continuation.yield(ChatResponseChunk.reasoning(reasoning))
                                    }
                                    if let content = choice.delta.content {
                                        if let toolProcessor {
                                            if let passthrough = toolProcessor.processChunk(content) {
                                                continuation.yield(ChatResponseChunk.text(passthrough))
                                            }
                                        } else {
                                            continuation.yield(ChatResponseChunk.text(content))
                                        }
                                    }
                                }
                            }

                            if decodeResult.shouldStop {
                                logger.info("reached an additional terminating token")
                                generationTask.cancel()
                                break generationLoop
                            }
                        }
                        await generationTask.value

                        let output = context.tokenizer.decode(tokenIds: generatedTokens)
                        if let finalChunk = decoder.makeChunk(
                            text: output,
                            previousLength: latestOutputLength,
                            shouldRemoveLeadingWhitespace: &shouldRemoveLeadingWhitespace
                        ) {
                            for choice in finalChunk.choices {
                                if let reasoning = choice.delta.reasoningContent {
                                    continuation.yield(ChatResponseChunk.reasoning(reasoning))
                                }
                                if let content = choice.delta.content {
                                    if let toolProcessor {
                                        if let passthrough = toolProcessor.processChunk(content) {
                                            continuation.yield(ChatResponseChunk.text(passthrough))
                                        }
                                    } else {
                                        continuation.yield(ChatResponseChunk.text(content))
                                    }
                                }
                            }
                        }
                        if let finalChunk = decoder.finalize(
                            shouldRemoveLeadingWhitespace: &shouldRemoveLeadingWhitespace
                        ) {
                            for choice in finalChunk.choices {
                                if let reasoning = choice.delta.reasoningContent {
                                    continuation.yield(.reasoning(reasoning))
                                }
                                if let content = choice.delta.content {
                                    if let toolProcessor {
                                        if let passthrough = toolProcessor.processChunk(content) {
                                            continuation.yield(.text(passthrough))
                                        }
                                    } else {
                                        continuation.yield(.text(content))
                                    }
                                }
                            }
                        }

                        if let toolProcessor {
                            for toolCall in toolProcessor.toolCalls {
                                let argsJSON = toolCallArgsToJSON(toolCall.function.arguments)
                                let request = ToolRequest(
                                    name: toolCall.function.name,
                                    args: argsJSON
                                )
                                continuation.yield(ChatResponseChunk.tool(request))
                            }
                        }

                        logger.info("inference completed, total output length: \(output.count)")
                        continuation.finish()
                    } catch is CancellationError {
                        logger.debug("inference cancelled for token: \(token.uuidString)")
                        continuation.finish()
                    } catch {
                        logger.error("inference failed: \(error.localizedDescription)")
                        continuation.finish(throwing: error)
                    }
                }

                continuation.onTermination = { @Sendable reason in
                    guard case .cancelled = reason else { return }
                    logger.debug("stream cancelled before completion for token: \(token.uuidString)")
                    workerTask.cancel()
                    MLXChatClientQueue.shared.release(token: token)
                }
            }
        }.eraseToAnyAsyncSequence()
    }
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
struct ChunkDecodeResult {
    let chunk: ChatCompletionChunk?
    let shouldStop: Bool
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
struct ChunkDecoder {
    let context: ModelContext
    var reasoningEmitter: ReasoningEventEmitter?

    init(context: ModelContext, input: LMInput) {
        self.context = context
        guard let config = context.configuration.reasoningConfig else { return }

        let promptTokens = input.text.tokens.asArray(Int.self)
        let prompt = context.tokenizer.decode(tokenIds: promptTokens)
        let primedInside = ReasoningEventEmitter.promptEndsInsideReasoning(
            renderedPromptTail: prompt,
            config: config
        )
        reasoningEmitter = ReasoningEventEmitter(
            config: config,
            primedInside: primedInside
        )
    }

    mutating func decode(
        tokens: [Int],
        latestOutputLength: inout Int,
        shouldRemoveLeadingWhitespace: inout Bool
    ) -> ChunkDecodeResult {
        var text = context.tokenizer.decode(tokenIds: tokens)
        let previousLength = latestOutputLength
        defer { latestOutputLength = text.count }

        while text.hasSuffix(MLXChatClient.decoderErrorSuffix) {
            text.removeLast(MLXChatClient.decoderErrorSuffix.count)
        }

        let chunk = makeChunk(
            text: text,
            previousLength: previousLength,
            shouldRemoveLeadingWhitespace: &shouldRemoveLeadingWhitespace
        )

        var mutableText = text
        var shouldStop = false
        for terminator in ChatClientConstants.additionalTerminatingTokens {
            var terminated = false
            while mutableText.hasSuffix(terminator) {
                mutableText.removeLast(terminator.count)
                terminated = true
            }
            if terminated {
                shouldStop = true
            }
        }

        return .init(chunk: chunk, shouldStop: shouldStop)
    }

    mutating func makeChunk(
        text: String,
        previousLength: Int,
        shouldRemoveLeadingWhitespace: inout Bool
    ) -> ChatCompletionChunk? {
        guard previousLength < text.count else { return nil }
        let chunkRange = previousLength ..< text.count
        let startIndex = text.index(text.startIndex, offsetBy: chunkRange.lowerBound)
        let endIndex = text.index(text.startIndex, offsetBy: chunkRange.upperBound)
        let chunkContent = String(text[startIndex ..< endIndex])

        let segments: [ReasoningEventEmitter.Segment]
        if var reasoningEmitter {
            segments = reasoningEmitter.process(chunkContent)
            self.reasoningEmitter = reasoningEmitter
        } else {
            segments = [.response(chunkContent)]
        }
        return makeChunk(
            segments: segments,
            shouldRemoveLeadingWhitespace: &shouldRemoveLeadingWhitespace
        )
    }

    mutating func finalize(
        shouldRemoveLeadingWhitespace: inout Bool
    ) -> ChatCompletionChunk? {
        guard var reasoningEmitter else { return nil }
        let segments = reasoningEmitter.finalize()
        self.reasoningEmitter = reasoningEmitter
        return makeChunk(
            segments: segments,
            shouldRemoveLeadingWhitespace: &shouldRemoveLeadingWhitespace
        )
    }

    private func makeChunk(
        segments: [ReasoningEventEmitter.Segment],
        shouldRemoveLeadingWhitespace: inout Bool
    ) -> ChatCompletionChunk? {
        let choices = segments.compactMap { segment -> ChatCompletionChunk.Choice? in
            var text: String
            let isReasoning: Bool
            switch segment {
            case let .reasoning(value):
                text = value
                isReasoning = true
            case let .response(value):
                text = value
                isReasoning = false
            }

            if shouldRemoveLeadingWhitespace {
                text = text.trimmingCharactersFromStart(in: .whitespacesAndNewlines)
                shouldRemoveLeadingWhitespace = text.isEmpty
            }
            guard !text.isEmpty else { return nil }

            let delta = if isReasoning {
                ChatCompletionChunk.Choice.Delta(reasoningContent: text)
            } else {
                ChatCompletionChunk.Choice.Delta(content: text)
            }
            return .init(delta: delta)
        }
        return choices.isEmpty ? nil : .init(choices: choices)
    }
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
private func toolCallArgsToJSON(_ arguments: [String: JSONValue]) -> String {
    let anyObject = arguments.mapValues { $0.anyValue }
    guard let data = try? JSONSerialization.data(withJSONObject: anyObject, options: [.sortedKeys]),
          let json = String(data: data, encoding: .utf8)
    else {
        return "{}"
    }
    return json
}
