import { extractModelAndProvider } from "./extractModelAndProvider";

/**
 * Given a list of models, organize them by provider
 * @param models The list of models
 * @returns An object containing the provider and models
 *
 * @example
 * const models = [
 *  "azure/ada",
 *  "azure/gpt-35-turbo",
 *  "cohere.command-r-v1:0",
 * ];
 *
 * organizeModelsAndProviders(models);
 * // returns {
 * //   azure: {
 * //     separator: "/",
 * //     models: ["ada", "gpt-35-turbo"],
 * //   },
 * //   cohere: {
 * //     separator: ".",
 * //     models: ["command-r-v1:0"],
 * //   },
 * // }
 */
export const organizeModelsAndProviders = (models: string[]) => {
  const object: Record<string, { separator: string; models: string[] }> = {};
  models.forEach((model) => {
    const {
      separator,
      provider,
      model: modelId,
    } = extractModelAndProvider(model);
    if (!object[provider]) {
      object[provider] = { separator, models: [] };
    }
    object[provider].models.push(modelId);
  });
  return object;
};