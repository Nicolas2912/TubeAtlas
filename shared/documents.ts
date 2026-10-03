import { formatTime, watchLink } from './time.ts';

export const kindLabels = { note: 'Note', summary: 'Summary', study_guide: 'Study guide', qa: 'Q&A', attachment: 'File' };
export type Format = 'bold' | 'italic' | 'bullet' | 'numbered' | 'quote' | 'paragraph' | 'h1' | 'h2' | 'h3' | 'link';
/** Returns the edited text and a selection to restore in the textarea. */
export function formatMarkdown(text: string, start: number, end: number, format: Format, url = 'https://') {
  if (['bold', 'italic', 'link'].includes(format)) {
    const selected = text.slice(start, end) || (format === 'link' ? 'link text' : 'text');
    const prefix = format === 'bold' ? '**' : format === 'italic' ? '*' : '[';
    const suffix = format === 'link' ? `](${url})` : prefix;
    return { text: text.slice(0, start) + prefix + selected + suffix + text.slice(end), start: start + prefix.length, end: start + prefix.length + selected.length };
  }
  const from = text.lastIndexOf('\n', Math.max(0, start - 1)) + 1;
  const next = text.indexOf('\n', end > start && text[end - 1] === '\n' ? end - 1 : end);
  const to = next < 0 ? text.length : next;
  const lines = text.slice(from, to).split('\n').map((line, index) => {
    if (format === 'bullet') return `- ${line}`;
    if (format === 'numbered') return `${index + 1}. ${line}`;
    if (format === 'quote') return `> ${line}`;
    const heading = format === 'paragraph' ? '' : '#'.repeat(Number(format.slice(1))) + ' ';
    return heading + line.replace(/^#{1,6}\s+/, '');
  }).join('\n');
  return { text: text.slice(0, from) + lines + text.slice(to), start: from, end: from + lines.length };
}
export function transcriptQuote(text: string, videoId: number, start: number | null) {
  const quote = `> "${text.trim().replace(/\n/g, '\n> ')}"`;
  return start === null ? quote : `${quote}\n> — [${formatTime(start)}](${watchLink(videoId, start)})`;
}
