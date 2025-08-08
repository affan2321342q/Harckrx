import { NextResponse } from "next/server";
import OpenAI from "openai";

// Create API client
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req) {
  try {
    const { query } = await req.json();

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

    return NextResponse.json(JSON.parse(completion.choices[0].message.content));
  } catch (error) {
    return NextResponse.json({ error: error.message });
  }
}
