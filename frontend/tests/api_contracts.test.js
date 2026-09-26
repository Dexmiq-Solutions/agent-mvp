import test from 'node:test';
import assert from 'node:assert/strict';

// Helper to simulate global fetch and ReadableStream
function setupMockFetch(responder) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    return responder(url, options);
  };
  return () => {
    globalThis.fetch = originalFetch;
  };
}

test('Projects API - listProjects calls GET /projects', async () => {
  let calledUrl = '';
  let calledMethod = '';

  const restore = setupMockFetch(async (url, options) => {
    calledUrl = String(url);
    calledMethod = options?.method || 'GET';
    return {
      ok: true,
      status: 200,
      json: async () => [{ id: 'proj-123', name: 'Test Project' }],
    };
  });

  try {
    const { listProjects } = await import('../src/api/projects.ts');
    const result = await listProjects();
    assert.equal(result.length, 1);
    assert.equal(result[0].id, 'proj-123');
    assert.match(calledUrl, /\/projects$/);
    assert.equal(calledMethod, 'GET');
  } finally {
    restore();
  }
});

test('Projects API - createProject posts payload to /projects', async () => {
  let requestBody = null;
  let requestMethod = '';

  const restore = setupMockFetch(async (url, options) => {
    requestMethod = options?.method;
    requestBody = JSON.parse(options?.body);
    return {
      ok: true,
      status: 201,
      json: async () => ({ id: 'proj-new', name: requestBody.name }),
    };
  });

  try {
    const { createProject } = await import('../src/api/projects.ts');
    const result = await createProject({ name: 'Alpha', description: 'Alpha Desc' });
    assert.equal(result.id, 'proj-new');
    assert.equal(result.name, 'Alpha');
    assert.equal(requestMethod, 'POST');
    assert.deepEqual(requestBody, { name: 'Alpha', description: 'Alpha Desc' });
  } finally {
    restore();
  }
});

test('Conversations API - listConversations and createConversation use project boundary', async () => {
  let recordedUrl = '';
  let recordedMethod = '';

  const restore = setupMockFetch(async (url, options) => {
    recordedUrl = String(url);
    recordedMethod = options?.method || 'GET';
    return {
      ok: true,
      status: 200,
      json: async () => [{ id: 'conv-1', project_id: 'proj-abc', title: 'Chat 1' }],
    };
  });

  try {
    const { listConversations, createConversation } = await import('../src/api/conversations.ts');
    const list = await listConversations('proj-abc');
    assert.equal(list.length, 1);
    assert.match(recordedUrl, /\/projects\/proj-abc\/conversations$/);
    assert.equal(recordedMethod, 'GET');

    await createConversation('proj-abc', 'My BRD Chat');
    assert.match(recordedUrl, /\/projects\/proj-abc\/conversations$/);
    assert.equal(recordedMethod, 'POST');
  } finally {
    restore();
  }
});

test('Conversations API - sendMessageStream handles SSE stream tokens and done event', async () => {
  const encoder = new TextEncoder();
  const chunks = [
    'event: message\ndata: {"type": "content", "content": "Hello "}\n\n',
    'event: message\ndata: {"type": "content", "content": "World!"}\n\n',
    'event: done\ndata: {"type": "done", "message": {"id": "msg-1", "role": "assistant", "content": "Hello World!"}}\n\n',
  ];

  const restore = setupMockFetch(async (url, options) => {
    assert.match(String(url), /\/projects\/p1\/conversations\/c1\/messages\?stream=true/);
    assert.equal(options.method, 'POST');

    const stream = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    });

    return {
      ok: true,
      status: 200,
      body: stream,
    };
  });

  try {
    const { sendMessageStream } = await import('../src/api/conversations.ts');

    const tokens = [];
    let completedMessage = null;

    await sendMessageStream({
      projectId: 'p1',
      conversationId: 'c1',
      content: 'Hi',
      onToken: (t) => tokens.push(t),
      onDone: (msg) => {
        completedMessage = msg;
      },
    });

    assert.equal(tokens.join(''), 'Hello World!');
    assert.equal(completedMessage?.id, 'msg-1');
    assert.equal(completedMessage?.content, 'Hello World!');
  } finally {
    restore();
  }
});

test('Sources API - listSources and uploadSource use project boundary', async () => {
  let uploadedFormData = null;
  let targetUrl = '';

  const restore = setupMockFetch(async (url, options) => {
    targetUrl = String(url);
    if (options?.body instanceof FormData) {
      uploadedFormData = options.body;
    }
    return {
      ok: true,
      status: 200,
      json: async () => [{ id: 'doc-1', project_id: 'proj-xyz', name: 'specs.pdf' }],
    };
  });

  try {
    const { listSources, uploadSource } = await import('../src/api/sources.ts');
    const sources = await listSources('proj-xyz');
    assert.equal(sources.length, 1);
    assert.match(targetUrl, /\/projects\/proj-xyz\/sources$/);

    const fakeFile = new Blob(['sample content'], { type: 'text/plain' });
    await uploadSource('proj-xyz', fakeFile, 'sample.txt');
    assert.match(targetUrl, /\/projects\/proj-xyz\/sources$/);
    assert.ok(uploadedFormData instanceof FormData);
  } finally {
    restore();
  }
});
