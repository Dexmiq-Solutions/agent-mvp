import test from 'node:test';
import assert from 'node:assert/strict';

function setupMockFetch(responder) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    return responder(url, options);
  };
  return () => {
    globalThis.fetch = originalFetch;
  };
}

test('Project Isolation - Requests strictly isolate project_id', async () => {
  const recordedEndpoints = [];

  const restore = setupMockFetch(async (url) => {
    recordedEndpoints.push(String(url));
    return {
      ok: true,
      status: 200,
      json: async () => [],
    };
  });

  try {
    const { listConversations, listMessages } = await import('../src/api/conversations.ts');
    const { listSources } = await import('../src/api/sources.ts');

    // Operations for Project 1
    await listConversations('proj-tenant-1');
    await listSources('proj-tenant-1');
    await listMessages('proj-tenant-1', 'conv-100');

    // Operations for Project 2
    await listConversations('proj-tenant-2');
    await listSources('proj-tenant-2');
    await listMessages('proj-tenant-2', 'conv-200');

    // Verify all URLs were strictly bound to their respective projects
    assert.equal(recordedEndpoints[0], '/projects/proj-tenant-1/conversations');
    assert.equal(recordedEndpoints[1], '/projects/proj-tenant-1/sources');
    assert.equal(recordedEndpoints[2], '/projects/proj-tenant-1/conversations/conv-100/messages');

    assert.equal(recordedEndpoints[3], '/projects/proj-tenant-2/conversations');
    assert.equal(recordedEndpoints[4], '/projects/proj-tenant-2/sources');
    assert.equal(recordedEndpoints[5], '/projects/proj-tenant-2/conversations/conv-200/messages');

    // Ensure no cross-project contamination
    for (let i = 0; i < 3; i++) {
      assert.ok(!recordedEndpoints[i].includes('proj-tenant-2'));
    }
    for (let i = 3; i < 6; i++) {
      assert.ok(!recordedEndpoints[i].includes('proj-tenant-1'));
    }
  } finally {
    restore();
  }
});

test('Error Handling - sendMessageStream surfaces server errors cleanly', async () => {
  const restore = setupMockFetch(async () => {
    return {
      ok: false,
      status: 500,
      json: async () => ({ detail: 'Agent processing pipeline failed due to LLM rate limit' }),
    };
  });

  try {
    const { sendMessageStream } = await import('../src/api/conversations.ts');

    let reportedError = null;
    await assert.rejects(async () => {
      await sendMessageStream({
        projectId: 'proj-error',
        conversationId: 'conv-error',
        content: 'Generate requirements',
        onError: (err) => {
          reportedError = err;
        },
      });
    }, /Agent processing pipeline failed due to LLM rate limit/);

    assert.equal(reportedError, 'Agent processing pipeline failed due to LLM rate limit');
  } finally {
    restore();
  }
});

test('Error Handling - SSE error event triggers onError callback', async () => {
  const encoder = new TextEncoder();
  const errorEvent = 'event: error\ndata: {"type": "error", "error": "RAG vector retrieval failed"}\n\n';

  const restore = setupMockFetch(async () => {
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(errorEvent));
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

    let errorReceived = null;
    await sendMessageStream({
      projectId: 'proj-1',
      conversationId: 'conv-1',
      content: 'Explain system architecture',
      onError: (err) => {
        errorReceived = err;
      },
    });

    assert.equal(errorReceived, 'RAG vector retrieval failed');
  } finally {
    restore();
  }
});
