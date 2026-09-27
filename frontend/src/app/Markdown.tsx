import ReactMarkdown from 'react-markdown';
import { Link } from 'react-router';

/** Renders Markdown with raw HTML disabled; app links stay in the app, external links open a new tab. */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        components={{
          a: ({ href = '', children: text }) =>
            href.startsWith('/') ? (
              <Link to={href}>{text}</Link>
            ) : (
              <a href={href} target="_blank" rel="noreferrer">
                {text}
              </a>
            ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
