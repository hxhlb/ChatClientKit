
import Foundation
import FoundationModels

@available(iOS 26.0, macOS 26, macCatalyst 26.0, *)
public class AppleIntelligenceChatClient: ChatService, @unchecked Sendable {
    public struct Configuration: Sendable {
        public var persona: String
        public var streamingPersona: String
        public var defaultTemperature: Double

        public init(
            persona: String = "",
            streamingPersona: String = "",
            defaultTemperature: Double = 0.75
        ) {
            self.persona = persona
            self.streamingPersona = streamingPersona
            self.defaultTemperature = defaultTemperature
        }
    }

    public let errorCollector = ErrorCollector.new()

    let configuration: Configuration

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
    }

    public func streamingChat(
        body: ChatRequestBody
    ) async throws -> AnyAsyncSequence<ChatResponseChunk> {
        try makeStreamingSequence(
            body: body,
            persona: configuration.streamingPersona
        )
    }

    struct SessionContext {
        let session: LanguageModelSession
        let prompt: String
        let options: GenerationOptions
    }

    func makeSessionContext(
        body: ChatRequestBody,
        persona: String
    ) throws -> SessionContext {
        let selectedTools = try selectTools(body.tools, choice: body.toolChoice)
        let tools = try makeToolProxies(from: selectedTools)

        let instructionText = AppleIntelligencePromptBuilder.makeInstructions(
            persona: persona,
            messages: body.messages,
            additionalDirectives: toolChoiceInstructions(body.toolChoice)
        )

        let prompt = AppleIntelligencePromptBuilder.makePrompt(from: body.messages)

        let session = if tools.isEmpty {
            LanguageModelSession(instructions: instructionText)
        } else {
            LanguageModelSession(
                tools: tools,
                instructions: instructionText
            )
        }

        let clampedTemperature = clampTemperature(
            body.temperature ?? configuration.defaultTemperature
        )
        let maximumResponseTokens = body.maxCompletionTokens.flatMap { value in
            value > 0 ? value : nil
        }
        var options = GenerationOptions(
            temperature: clampedTemperature,
            maximumResponseTokens: maximumResponseTokens
        )
        if #available(iOS 27.0, macOS 27.0, macCatalyst 27.0, *) {
            switch body.toolChoice {
            case .required, .function:
                options.toolCallingMode = .required
            case .auto:
                options.toolCallingMode = tools.isEmpty ? nil : .allowed
            case nil:
                break
            }
        }

        return SessionContext(session: session, prompt: prompt, options: options)
    }

    func selectTools(
        _ tools: [ChatRequestBody.Tool]?,
        choice: ChatRequestBody.ToolChoice?
    ) throws -> [ChatRequestBody.Tool] {
        let tools = tools ?? []
        guard case let .function(selectedName) = choice else {
            if case .required = choice, tools.isEmpty {
                throw invalidToolChoiceError(
                    String(localized: "A required tool choice needs at least one tool.")
                )
            }
            return tools
        }

        guard let selected = tools.first(where: { tool in
            guard case let .function(name, _, _, _) = tool else { return false }
            return name == selectedName
        }) else {
            let key: String.LocalizationValue = "The selected tool '\(selectedName)' is not available."
            throw invalidToolChoiceError(String(localized: key))
        }
        return [selected]
    }

    func makeToolProxies(
        from tools: [ChatRequestBody.Tool]
    ) throws -> [any Tool] {
        try tools.map { tool -> any Tool in
            switch tool {
            case let .function(name, description, parameters, _):
                return try AppleIntelligenceToolProxy(
                    name: name,
                    description: description,
                    parameters: parameters
                ) as any Tool
            }
        }
    }

    func toolChoiceInstructions(_ choice: ChatRequestBody.ToolChoice?) -> [String] {
        switch choice {
        case .required:
            ["Call one of the available tools before answering the user."]
        case let .function(name):
            ["Call the \(name) tool. Do not answer the user directly."]
        case .auto, nil:
            []
        }
    }

    func clampTemperature(_ value: Double) -> Double {
        let fallback = configuration.defaultTemperature.isFinite
            ? configuration.defaultTemperature
            : 0.75
        guard value.isFinite else { return min(max(fallback, 0), 2) }
        return min(max(value, 0), 2)
    }

    private func invalidToolChoiceError(_ description: String) -> NSError {
        NSError(
            domain: "AppleIntelligence",
            code: -2,
            userInfo: [NSLocalizedDescriptionKey: description]
        )
    }

    func makeStreamingSequence(
        body: ChatRequestBody,
        persona: String
    ) throws -> AnyAsyncSequence<ChatResponseChunk> {
        guard AppleIntelligenceModel.shared.isAvailable else {
            throw NSError(
                domain: "AppleIntelligence",
                code: -1,
                userInfo: [
                    NSLocalizedDescriptionKey: String(localized: "Apple Intelligence is not available."),
                ]
            )
        }

        let context = try makeSessionContext(
            body: body,
            persona: persona
        )

        return AnyAsyncSequence(AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var accumulated = ""
                    for try await partial in context.session.streamResponse(
                        to: context.prompt,
                        options: context.options
                    ) {
                        let fullText = partial.content
                        guard fullText.count >= accumulated.count else {
                            accumulated = ""
                            continue
                        }

                        let deltaStart = fullText.index(fullText.startIndex, offsetBy: accumulated.count)
                        let newContent = String(fullText[deltaStart...])
                        accumulated = fullText

                        guard !newContent.isEmpty else { continue }

                        continuation.yield(.text(newContent))
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch let error as LanguageModelSession.ToolCallError {
                    guard let invocationError = error.underlyingError as? AppleIntelligenceToolError else {
                        continuation.finish(throwing: error)
                        return
                    }
                    switch invocationError {
                    case let .invocationCaptured(request):
                        continuation.yield(.tool(request))
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        })
    }
}
