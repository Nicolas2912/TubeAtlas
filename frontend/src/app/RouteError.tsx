import { isRouteErrorResponse, Link, useRouteError } from 'react-router';

export function RouteError() {
  const error = useRouteError();
  const notFound = isRouteErrorResponse(error) && error.status === 404;
  return (
    <div className="empty">
      <h1>{notFound ? 'Page not found' : 'Something went wrong'}</h1>
      <p className="muted">{notFound ? 'This page does not exist.' : error instanceof Error ? error.message : 'An unexpected error occurred.'}</p>
      <Link to="/" className="button">
        Back to the library
      </Link>
    </div>
  );
}
