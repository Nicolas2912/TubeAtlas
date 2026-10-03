import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router';
import { Shell } from './Shell.tsx';
import { RouteError } from './RouteError.tsx';

const Library = lazy(() => import('../features/library/LibraryPage.tsx'));
const Topics = lazy(() => import('../features/topics/TopicsPage.tsx'));
const Settings = lazy(() => import('../features/settings/SettingsPage.tsx'));
const Video = lazy(() => import('./VideoLayout.tsx'));
const VideoOverview = lazy(() => import('./VideoLayout.tsx').then((module) => ({ default: module.VideoOverview })));

const router = createBrowserRouter([
  {
    element: <Shell />,
    errorElement: <RouteError />,
    hydrateFallbackElement: <p className="notice" role="status">Loading TubeAtlas…</p>,
    children: [
      { index: true, element: <Library /> },
      { path: 'topics', element: <Topics /> },
      { path: 'topics/:topicId', element: <Topics /> },
      { path: 'settings', element: <Settings /> },
      { path: 'videos/:videoId', element: <Video />, children: [{ index: true, element: <VideoOverview /> }] },
      { path: '*', element: <RouteError />, errorElement: <RouteError />, loader: () => { throw new Response('Not found', { status: 404 }); } },
    ],
  },
]);

export default function App() {
  return <Suspense fallback={<p className="notice" role="status">Loading TubeAtlas…</p>}><RouterProvider router={router} /></Suspense>;
}
