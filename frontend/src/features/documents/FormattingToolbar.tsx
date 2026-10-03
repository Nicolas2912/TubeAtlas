import type { RefObject } from 'react';
import { formatMarkdown, type Format } from '../../../../shared/documents.ts';

export function FormattingToolbar({ textarea, text, onChange }: { textarea: RefObject<HTMLTextAreaElement | null>; text: string; onChange: (text: string) => void }) {
  function apply(format: Format) {
    const field = textarea.current;
    if (!field) return;
    const url = format === 'link' ? window.prompt('Link address', 'https://') : undefined;
    if (url === null) return;
    const result = formatMarkdown(text, field.selectionStart, field.selectionEnd, format, url);
    onChange(result.text);
    requestAnimationFrame(() => { field.focus(); field.setSelectionRange(result.start, result.end); });
  }
  return <div className="formatting-toolbar" role="group" aria-label="Text formatting">
    <select className="input" aria-label="Paragraph style" value="" onChange={(e) => apply(e.target.value as Format)}><option value="" disabled>Paragraph style</option><option value="paragraph">Paragraph</option><option value="h1">Heading 1</option><option value="h2">Heading 2</option><option value="h3">Heading 3</option></select>
    {(['bold', 'italic', 'bullet', 'numbered', 'link', 'quote'] as const).map((format) => <button key={format} className={`button small format-${format}`} aria-label={{ bold: 'Bold', italic: 'Italic', bullet: 'Bullet list', numbered: 'Numbered list', link: 'Link', quote: 'Quote' }[format]} onMouseDown={(e) => e.preventDefault()} onClick={() => apply(format)}>{{ bold: 'B', italic: 'I', bullet: '• List', numbered: '1. List', link: 'Link', quote: '“' }[format]}</button>)}
  </div>;
}
