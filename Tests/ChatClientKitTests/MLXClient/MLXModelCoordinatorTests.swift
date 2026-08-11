//
//  MLXModelCoordinatorTests.swift
//  ChatClientKitTests
//
//  Created by GPT-5 Codex on 2025/11/10.
//

@testable import ChatClientKit
import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLLM
@preconcurrency import MLXLMCommon
@preconcurrency import MLXVLM
import Testing

@Suite(.serialized)
struct MLXModelCoordinatorTests {
    @Test
    func `Default tokenizer loader resolves tokenizer from local fixture`() async throws {
        guard #available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *) else { return }

        let directory = try #require(TestFixtures.tokenizerOnlyModelDirectory())
        let loader = DefaultMLXTokenizerLoader()

        let tokenizer = try await loader.load(from: directory)

        #expect(tokenizer.convertTokenToId("offline") == 4)
        #expect(tokenizer.decode(tokenIds: [4, 5], skipSpecialTokens: false) == "offline path")
    }

    @Test
    func `Coordinator caches containers for identical configuration and kind`() async throws {
        guard #available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *) else { return }
        guard TestFixtures.mlxTestingModelDirectory() != nil else { return }

        let config = try modelConfiguration()
        let coordinator = MLXModelCoordinator(tokenizerLoader: StubTokenizerLoader())

        let first = try await coordinator.container(for: config, kind: .llm)
        let second = try await coordinator.container(for: config, kind: .llm)

        #expect(first === second)
    }

    @Test
    func `Coordinator reuses in-flight task for identical concurrent requests`() async throws {
        guard #available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *) else { return }
        guard TestFixtures.mlxTestingModelDirectory() != nil else { return }

        let config = try modelConfiguration()
        let coordinator = MLXModelCoordinator(tokenizerLoader: StubTokenizerLoader())

        async let pendingFirst = coordinator.container(for: config, kind: .llm)
        async let pendingSecond = coordinator.container(for: config, kind: .llm)

        let containers = try await (pendingFirst, pendingSecond)
        #expect(containers.0 === containers.1)
    }

    @Test
    func `Reset clears cached container`() async throws {
        guard #available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *) else { return }
        guard TestFixtures.mlxTestingModelDirectory() != nil else { return }

        let config = try modelConfiguration()
        let coordinator = MLXModelCoordinator(tokenizerLoader: StubTokenizerLoader())

        let first = try await coordinator.container(for: config, kind: .llm)
        await coordinator.reset()
        let second = try await coordinator.container(for: config, kind: .llm)

        #expect(first !== second)
    }
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
func modelConfiguration() throws -> ModelConfiguration {
    guard let url = TestFixtures.mlxTestingModelDirectory() else {
        fatalError("Fixture mlx_testing_model not found. Expected at ~/.testing/mlx_testing_model or <repo>/.test/mlx_testing_model.")
    }
    return ModelConfiguration(directory: url)
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
private struct StubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        _ = directory
        return StubTokenizer()
    }
}

@available(iOS 17.0, macOS 14.0, macCatalyst 17.0, *)
private struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        _ = text
        _ = addSpecialTokens
        return []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        _ = tokenIds
        _ = skipSpecialTokens
        return ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        _ = token
        return nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        _ = id
        return nil
    }

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        nil
    }

    var unknownToken: String? {
        nil
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        _ = messages
        _ = tools
        _ = additionalContext
        return []
    }
}

private enum TestFixtures {
    static func tokenizerOnlyModelDirectory() -> URL? {
        Bundle.module.url(
            forResource: "TokenizerOnlyModel",
            withExtension: nil,
            subdirectory: "Fixtures"
        )
    }

    static func mlxTestingModelDirectory(file: StaticString = #filePath) -> URL? {
        #if os(macOS)
            let homeFixture = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".testing")
                .appendingPathComponent("mlx_testing_model", isDirectory: true)
        #else
            let homeFixture = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)
                .first!
                .appendingPathComponent(".testing")
                .appendingPathComponent("mlx_testing_model", isDirectory: true)
        #endif

        if FileManager.default.fileExists(atPath: homeFixture.path) {
            return homeFixture
        }

        var url = URL(fileURLWithPath: "\(file)")
        for _ in 0 ..< 5 {
            url.deleteLastPathComponent()
        }

        let repoFixture = url
            .appendingPathComponent(".test")
            .appendingPathComponent("mlx_testing_model", isDirectory: true)

        guard FileManager.default.fileExists(atPath: repoFixture.path) else {
            return nil
        }
        return repoFixture
    }
}
