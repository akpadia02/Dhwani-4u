import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI("AIzaSyC5jSD_-qIoNf_FU41pN0WCN1rSaPnovQk");

export async function detectIntentFromText(text) {
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });

  try {
    const prompt = `What is the speaker's intent in the following sentence? Respond with only 1-2 words.\n\n"${text}"`;

    const result = await model.generateContent([prompt]);
    const response = await result.response;
    return response.text().trim();
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw error;
  }
}
