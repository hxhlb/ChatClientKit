import Foundation
import FoundationModels

@available(iOS 26.0, macOS 26, macCatalyst 26.0, *)
enum AppleIntelligenceToolError: Error {
    case invocationCaptured(ToolRequest)
}

@available(iOS 26.0, macOS 26, macCatalyst 26.0, *)
struct AppleIntelligenceToolProxy: Tool {
    let name: String
    let description: String
    let parameters: GenerationSchema

    init(
        name: String,
        description: String?,
        parameters: [String: AnyCodingValue]?
    ) throws {
        self.name = name
        self.description = description?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        self.parameters = try AppleIntelligenceToolSchema.makeGenerationSchema(
            from: parameters
        )
    }

    func call(arguments: GeneratedContent) async throws -> String {
        let request = ToolRequest(name: name, args: arguments.jsonString)
        throw AppleIntelligenceToolError.invocationCaptured(request)
    }
}

@available(iOS 26.0, macOS 26, macCatalyst 26.0, *)
private enum AppleIntelligenceToolSchema {
    static func makeGenerationSchema(
        from parameters: [String: AnyCodingValue]?
    ) throws -> GenerationSchema {
        let schema = normalize(
            parameters?.untypedDictionary ?? [
                "type": "object",
                "properties": [String: Any](),
            ],
            title: "ToolArguments"
        )
        let data = try JSONSerialization.data(withJSONObject: schema, options: [.sortedKeys])
        return try JSONDecoder().decode(GenerationSchema.self, from: data)
    }

    private static func normalize(_ value: Any, title: String) -> Any {
        if let array = value as? [Any] {
            return array.enumerated().map { index, item in
                normalize(item, title: "\(title)Item\(index)")
            }
        }

        guard var object = value as? [String: Any] else { return value }

        if let properties = object["properties"] as? [String: Any] {
            let names = properties.keys.sorted()
            object["properties"] = Dictionary(uniqueKeysWithValues: names.map { name in
                (name, normalize(properties[name]!, title: "\(title)_\(name)"))
            })
            object["x-order"] = names
            object["required"] = object["required"] ?? [String]()
        }

        if let items = object["items"] {
            object["items"] = normalize(items, title: "\(title)Item")
        }

        if object["type"] as? String == "object" || object["properties"] != nil {
            object["type"] = "object"
            object["title"] = object["title"] ?? title
            object["properties"] = object["properties"] ?? [String: Any]()
            object["x-order"] = object["x-order"] ?? [String]()
            object["required"] = object["required"] ?? [String]()
            object["additionalProperties"] = false
        }

        return object.mapValues { normalize($0, title: title) }
    }
}
