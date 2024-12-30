import { Hono } from 'hono'


const app = new Hono<{ Bindings: CloudflareBindings }>()

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

app.get('/health', (c) => {
  return c.json({ code: 200, status: 'healthy' })
})

app.post('/vector/:userId/:semester/:courseId', async (c) => {
  const userId = c.req.param('userId')
  const semester = c.req.param('semester')
  const courseId = c.req.param('courseId')
  const { filename, query, action } = await c.req.json()
  
  const { data } = await c.env.AI.run('@cf/baai/bge-base-en-v1.5', {text: query})
  
  const SIMILARITY_THRESHOLD = 0.49

  let simmatches
  if (action === 'courseaction') {
    simmatches = await c.env.VECTORIZE.query(data[0], {
      topK: 7,
      filter: { user: userId, semester: semester, course: courseId },
      returnValues: true,
      returnMetadata: 'all',
    } )
  } else if (action === 'fileaction') {
    simmatches = await c.env.VECTORIZE.query(data[0], {
      topK: 7,
      filter: {user: userId, semester: semester, course: courseId, filename: filename},
      returnValues: true,
      returnMetadata: 'all',
    } )
  } else {
    throw new Error('Invalid action type')
  }

  const vecResults = simmatches.matches.filter(vec => vec.score > SIMILARITY_THRESHOLD).map(vec => ({
    id: vec.id,
    title: vec.metadata?.title,
  }))

  return c.json({ matches: vecResults })
})

app.post('/llama/summarizetopic', async (c) => {
  const { topic, context } = await c.req.json()
  console.log(topic, context);
  const messages = [
    { role: "system", content: "You are a highly skilled summarizer specializing in generating detailed, topic-specific summaries based solely on the provided context. Your goal is to extract the most relevant and comprehensive information related to the given topic while ensuring the summary is clear, accurate, and concise. Follow these instructions carefully: 1. **Understand the Topic**: - Focus entirely on the topic provided and disregard unrelated details. - Ensure the summary is fully aligned with the topic, reflecting the depth and scope of the context. 2. **Analyze the Context Step by Step**: - Slowly and methodically read the context text provided. - Identify all relevant points, facts, and examples that directly support the topic. 3. **Structure the Summary**: - Organize the information logically and cohesively to create a clear narrative. - Avoid redundancy by summarizing similar points into concise sentences. 4. **Focus on Clarity and Simplicity**: - Use simple language to make the summary accessible and easy to understand. - Avoid technical jargon unless it is essential for explaining the topic. 5. **Be Detailed but Concise**: - Include all critical details necessary to fully explain the topic. - Keep the response concise by omitting unnecessary elaboration or repetition. 6. **Do Not Add Unverified Information**: - Use only the information present in the provided context. - If the context lacks sufficient information to generate a meaningful summary, clearly indicate that the context is insufficient. 7. **Format Your Response**: - Write the summary as a plain text paragraph without any headings, introductions, or concluding statements. - Directly respond with the summary, formatted as a standalone paragraph. ### Example of Input and Output: Input: <context> <title>Economic Principles</title> <text>The principle of opportunity cost is central to economics. It refers to the value of the next best alternative that must be forgone when making a choice...</text> </context> Topic: Opportunity Cost Output: The principle of opportunity cost refers to the value of the next best alternative forgone when making a decision. It highlights the trade-offs involved in allocating limited resources and is fundamental to economic decision-making. For example, choosing to spend money on a vacation means giving up the opportunity to invest that money or use it for other needs. Opportunity cost applies in various contexts, such as individual choices, business decisions, and government policies." },
    {
      role: "user",
      content: `(Topic: ${topic})\n(Context: ${context})`,
    },
  ];

  const response = await c.env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", { messages });

  return Response.json(response);

})

export default app