import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { 
  Youtube, 
  Brain, 
  BarChart3, 
  Search, 
  Network, 
  Zap,
  ArrowRight,
  Play,
  TrendingUp,
  Eye
} from 'lucide-react';

const HomePage = () => {
  const features = [
    {
      icon: Youtube,
      title: 'YouTube Integration',
      description: 'Seamlessly extract transcripts and metadata from any YouTube channel',
      color: 'text-red-500'
    },
    {
      icon: Brain,
      title: 'AI-Powered Analysis',
      description: 'Advanced LLM processing to extract meaningful insights from video content',
      color: 'text-purple-500'
    },
    {
      icon: Network,
      title: 'Knowledge Graphs',
      description: 'Build comprehensive knowledge graphs showing relationships between concepts',
      color: 'text-blue-500'
    },
    {
      icon: Search,
      title: 'Smart Search',
      description: 'Query your knowledge base with natural language and get instant answers',
      color: 'text-green-500'
    },
    {
      icon: BarChart3,
      title: 'Deep Insights',
      description: 'Uncover patterns, themes, and hidden connections in video content',
      color: 'text-orange-500'
    },
    {
      icon: Zap,
      title: 'Fast Processing',
      description: 'Efficient batch processing with OpenAI API for large-scale analysis',
      color: 'text-yellow-500'
    }
  ];

  const stats = [
    { label: 'Channels Analyzed', value: '1,000+' },
    { label: 'Hours Processed', value: '50,000+' },
    { label: 'Insights Generated', value: '25,000+' },
    { label: 'Knowledge Graphs', value: '500+' }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary-50 via-white to-blue-50">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mb-8"
            >
              <div className="inline-flex items-center space-x-2 bg-white/80 backdrop-blur-sm rounded-full px-4 py-2 border border-gray-200 mb-6">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                <span className="text-sm text-gray-600">Now with GPT-4 Integration</span>
              </div>
              
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-gray-900 mb-6">
                Unlock the Power of
                <span className="block gradient-text">YouTube Content</span>
              </h1>
              
              <p className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                Transform YouTube channels into actionable insights with AI-powered analysis, 
                knowledge graphs, and deep content understanding.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4"
            >
              <Link
                to="/analyze"
                className="inline-flex items-center space-x-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold px-8 py-4 rounded-xl transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <Play className="w-5 h-5" />
                <span>Start Analyzing</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              
              <Link
                to="/insights"
                className="inline-flex items-center space-x-2 bg-white hover:bg-gray-50 text-gray-900 font-semibold px-8 py-4 rounded-xl border border-gray-200 transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
              >
                <Eye className="w-5 h-5" />
                <span>View Demo</span>
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-white py-16 border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600 font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                Everything you need to analyze YouTube content
              </h2>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                From transcript extraction to knowledge graph generation, TubeAtlas provides 
                a complete toolkit for understanding video content at scale.
              </p>
            </motion.div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="card feature-card group cursor-pointer"
                >
                  <div className={`w-12 h-12 rounded-lg bg-gray-100 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-200`}>
                    <Icon className={`w-6 h-6 ${feature.color}`} />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-primary-600 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Ready to unlock insights from YouTube?
            </h2>
            <p className="text-xl text-primary-100 mb-8 max-w-2xl mx-auto">
              Join thousands of researchers, content creators, and analysts who use TubeAtlas 
              to understand video content like never before.
            </p>
            <Link
              to="/analyze"
              className="inline-flex items-center space-x-2 bg-white hover:bg-gray-100 text-primary-600 font-semibold px-8 py-4 rounded-xl transition-all duration-200 transform hover:scale-105 shadow-lg hover:shadow-xl"
            >
              <TrendingUp className="w-5 h-5" />
              <span>Get Started for Free</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default HomePage; 