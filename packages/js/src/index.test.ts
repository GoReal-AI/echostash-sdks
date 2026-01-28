import { describe, it, expect } from 'vitest';
import {
  Echostash,
  LoadedPrompt,
  toOpenAI,
  toAnthropic,
  toGoogle,
  toVercel,
  toLangChain,
} from './index.js';

describe('Echostash SDK', () => {
  describe('LoadedPrompt', () => {
    it('should substitute variables', () => {
      const prompt = new LoadedPrompt({
        id: 'test',
        content: 'Hello {{name}}!',
        meta: {},
        parameterSymbol: '{{}}',
      });

      const rendered = prompt.with({ name: 'Alice' });
      expect(rendered.text()).toBe('Hello Alice!');
    });

    it('should keep placeholders for missing variables', () => {
      const prompt = new LoadedPrompt({
        id: 'test',
        content: 'Hello {{name}}!',
        meta: {},
        parameterSymbol: '{{}}',
      });

      const rendered = prompt.with({});
      // Unsubstituted variables remain as-is
      expect(rendered.text()).toBe('Hello {{name}}!');
    });

    it('should convert to OpenAI format', () => {
      const prompt = new LoadedPrompt({
        id: 'test',
        content: 'Hello world',
        meta: {},
      });

      const msg = prompt.openai({ role: 'system' });
      expect(msg).toEqual({ role: 'system', content: 'Hello world' });
    });

    it('should convert to Anthropic format', () => {
      const prompt = new LoadedPrompt({
        id: 'test',
        content: 'Hello world',
        meta: {},
      });

      const msg = prompt.anthropic({ role: 'user' });
      expect(msg).toEqual({ role: 'user', content: 'Hello world' });
    });
  });

  describe('Provider Converters', () => {
    it('toOpenAI should handle string content', () => {
      const msg = toOpenAI('Hello world', { role: 'user' });
      expect(msg).toEqual({ role: 'user', content: 'Hello world' });
    });

    it('toOpenAI should handle multi-modal content', () => {
      const content = [
        { type: 'text' as const, text: 'Describe this:' },
        { type: 'image_url' as const, image_url: { url: 'https://example.com/img.png' } },
      ];
      const msg = toOpenAI(content, { role: 'user' });
      expect(msg.role).toBe('user');
      expect(Array.isArray(msg.content)).toBe(true);
    });

    it('toAnthropic should handle string content', () => {
      const msg = toAnthropic('Hello world', { role: 'user' });
      expect(msg).toEqual({ role: 'user', content: 'Hello world' });
    });

    it('toGoogle should handle string content', () => {
      const msg = toGoogle('Hello world', { role: 'user' });
      expect(msg).toEqual({ role: 'user', parts: [{ text: 'Hello world' }] });
    });

    it('toVercel should handle string content', () => {
      const msg = toVercel('Hello world', { role: 'system' });
      expect(msg).toEqual({ role: 'system', content: 'Hello world' });
    });

    it('toLangChain should handle string content', () => {
      const msg = toLangChain('Hello world', { type: 'human' });
      expect(msg).toEqual({ type: 'human', content: 'Hello world' });
    });
  });

  describe('Echostash Client', () => {
    it('should create a client', () => {
      const client = new Echostash('https://api.example.com', {
        apiKey: 'test-key',
      });
      expect(client).toBeInstanceOf(Echostash);
    });

    it('should create a PromptQuery', () => {
      const client = new Echostash('https://api.example.com');
      const query = client.prompt('test-prompt');
      expect(query).toBeDefined();
    });
  });
});
