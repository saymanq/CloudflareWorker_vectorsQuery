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

app.post('/llama/summarizecourse', async (c) => {
  const { summaryJSON } = await c.req.json()
  console.log(summaryJSON);
  const messages = [
    { role: "system", content: "You are a formatting and summarization expert. Your task is to combine an array of topics and their corresponding summaries into a single, cohesive, and well-structured document formatted strictly in Markdown. Follow these instructions carefully: 1. **Understand the Input Format**: - The input will be an array of objects, each containing: - `topic`: The title of a specific topic. - `summary`: The detailed explanation for that topic. 2. **Adhere to the Provided Information**: - Use only the information given in the topics and summaries. - Do not introduce new content, expand beyond the provided summaries, or make assumptions. 3. **Generate a Cohesive Document**: - Arrange each topic as a Markdown heading, using the topic as the heading text. - Follow each heading with its corresponding summary. - Ensure the transitions between sections are seamless so the document reads as one unified summary of the entire file. 4. **Formatting Guidelines**: - Use Markdown syntax: - Use `#` for top-level headings (e.g., topics). - Use bullet points, numbered lists, or subheadings (`##`, `###`) within summaries if required for clarity. - Maintain consistent formatting and indentation throughout the document. - Write summaries in clear and concise paragraphs, ensuring readability. 5. **Output Requirements**: - The document must be well-structured, easy to read, and properly formatted in Markdown. - Do not include any introductory or concluding statements outside the Markdown structure. - Ensure the document remains focused on the provided topics and summaries without adding new information. ### Example Input: [ { topic: \"Introduction to Economics\", summary: \"Economics is the study of how people allocate scarce resources...\" }, { topic: \"Opportunity Cost\", summary: \"The concept of opportunity cost highlights the trade-offs involved in decision-making...\" } ]; Output: # Introduction to Economics Economics is the study of how people allocate scarce resources. It involves understanding how individuals, businesses, and governments make decisions about production, distribution, and consumption. Key principles include supply and demand, market structures, and the role of incentives. # Opportunity Cost The concept of opportunity cost highlights the trade-offs involved in decision-making. It refers to the value of the next best alternative foregone when a choice is made. This principle is essential in understanding how resources are prioritized and utilized in various scenarios." },
    {
      role: "user",
      content: `${JSON.stringify(summaryJSON)}`,
    },
  ];

  const response = await c.env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", { messages });

  return Response.json(response);

})

export default app