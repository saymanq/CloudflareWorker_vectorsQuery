import { Hono } from 'hono'


const app = new Hono<{ Bindings: CloudflareBindings }>()

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

app.get('/health', (c) => {
  return c.json({ code: 200, status: 'healthy' })
})

app.post('/:userId/:semester/:courseId', async (c) => {
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

  const vecIds = simmatches.matches.filter(vec => vec.score > SIMILARITY_THRESHOLD).map(vec => vec.id)

  return c.json({ vecIds })
})

export default app