import { NextResponse } from "next/server";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { query } = req.body;

    const prompt = `
      You are an insurance claims decision assistant.
      Given the query: "${query}",
      - Identify details (age, procedure, location, policy duration)
      - Check coverage from general insurance policy rules.
      - Return JSON with:
        {
          "decision": "approved/rejected",
          "amount": number or null,
          "justification": "reason with clause reference"
        }
    `;

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
    });

    res.status(200).json(JSON.parse(completion.choices[0].message.content));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
