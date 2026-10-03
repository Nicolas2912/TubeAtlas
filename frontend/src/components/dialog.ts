import type { KeyboardEvent } from 'react';

/** Keep Tab and Shift+Tab inside a modal, including when browser chrome would take focus. */
export function trapDialogFocus(event: KeyboardEvent<HTMLDialogElement>) {
  if (event.key !== 'Tab') return;
  const controls = [...event.currentTarget.querySelectorAll<HTMLElement>('button, input, select, textarea, a[href], [tabindex]')]
    .filter((el) => el.tabIndex >= 0 && !el.matches(':disabled') && el.getClientRects().length > 0);
  const target = event.shiftKey && document.activeElement === controls[0]
    ? controls.at(-1)
    : !event.shiftKey && document.activeElement === controls.at(-1) ? controls[0] : undefined;
  if (target) { event.preventDefault(); target.focus(); }
}
