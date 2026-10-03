/** ASCII fallback plus the UTF-8 filename; neither can inject headers or paths. */
export function contentDisposition(mode: 'inline' | 'attachment', name: string) {
  const safe = name.replace(/[\u0000-\u001f\u007f/\\:*?"<>|]/g, '_').slice(0, 180) || 'download';
  const ascii = safe.replace(/[^\x20-\x7e]/g, '_');
  const encoded = encodeURIComponent(safe).replace(/['()*]/g, (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
  return `${mode}; filename="${ascii}"; filename*=UTF-8''${encoded}`;
}
