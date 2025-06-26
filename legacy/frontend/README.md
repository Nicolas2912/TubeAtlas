# TubeAtlas Frontend

A modern React.js frontend for TubeAtlas - YouTube channel analysis and insights platform.

## 🚀 Features

- **Modern UI/UX**: Clean, responsive design with Tailwind CSS
- **Interactive Analysis**: Real-time progress tracking for channel analysis
- **Knowledge Graphs**: Visualize relationships between concepts and topics
- **Insights Dashboard**: Comprehensive view of channel insights and themes
- **Mobile Responsive**: Optimized for all device sizes
- **Smooth Animations**: Enhanced user experience with Framer Motion

## 🛠️ Tech Stack

- **React 19** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful, customizable icons

## 📦 Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and visit `http://localhost:5173`

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/           # Reusable components
│   │   ├── layout/          # Layout components (Header, Footer)
│   │   ├── ui/              # UI components (Button, Card, etc.)
│   │   └── features/        # Feature-specific components
│   ├── pages/               # Page components
│   │   ├── HomePage.jsx     # Landing page
│   │   ├── AnalyzePage.jsx  # Channel analysis interface
│   │   └── InsightsPage.jsx # Results and insights dashboard
│   ├── hooks/               # Custom React hooks
│   ├── utils/               # Utility functions
│   ├── services/            # API services
│   ├── contexts/            # React contexts
│   ├── App.jsx              # Main app component
│   └── main.jsx             # App entry point
├── public/                  # Static assets
└── package.json
```

## 🎨 Design System

### Colors
- **Primary**: Blue theme (`#3b82f6`)
- **Secondary**: Gray theme
- **Accent**: Various colors for different features

### Components
- **Button**: Multiple variants (primary, secondary, outline, etc.)
- **Card**: Flexible card component with animations
- **Layout**: Responsive header and footer

### Typography
- **Font**: Inter (Google Fonts)
- **Scales**: Tailwind's default typography scale

## 🔧 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 🚀 Deployment

1. Build the project:
   ```bash
   npm run build
   ```

2. Deploy the `dist` folder to your hosting platform

## 🔗 Integration with Backend

The frontend is designed to integrate with the TubeAtlas Python backend:

- **Analysis API**: Trigger channel analysis
- **Results API**: Fetch insights and knowledge graphs
- **Data Export**: Download analysis results

## 📱 Responsive Design

The application is fully responsive and optimized for:
- **Desktop**: Full-featured experience
- **Tablet**: Adapted layouts and navigation
- **Mobile**: Touch-friendly interface with collapsible menus

## 🎯 Key Pages

### Home Page
- Hero section with call-to-action
- Feature showcase
- Statistics and testimonials

### Analyze Page
- Channel URL input
- Real-time progress tracking
- Example channels for quick testing

### Insights Page
- Multi-tab interface (Overview, Insights, Knowledge Graph, etc.)
- Interactive data visualization
- Export and sharing capabilities

## 🔮 Future Enhancements

- Real-time WebSocket connections for live updates
- Advanced knowledge graph visualization with D3.js
- User authentication and saved analyses
- Collaborative features for team analysis
- Advanced filtering and search capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of TubeAtlas and follows the same licensing terms.
