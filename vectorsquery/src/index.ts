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
    { role: "system", content: "You are a perfect summarizer, you are given a topic and you need to find relevant information only from the context provided and create a summary in very simple terms and words only related to the topic from the context. Respond as if you are directly giving the answer and do not include any introductory or ending sentences (for example, sure, I can provide an answer). Be straight to the point." },
    {
      role: "user",
      content: `(Topic: ${topic})\n(Context: ${context})`,
    },
  ];

  const response = await c.env.AI.run("@cf/meta/llama-3.3-70b-instruct-fp8-fast", { messages });

  return Response.json(response);

})

export default app