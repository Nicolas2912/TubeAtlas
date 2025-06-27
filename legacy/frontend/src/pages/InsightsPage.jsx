import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Brain,
  Network,
  Download,
  Share2,
  Search,
  Filter,
  BarChart3,
  Eye,
  FileText,
  TrendingUp,
  MessageSquare,
  Clock,
  Tag
} from 'lucide-react';

const InsightsPage = () => {
  const [selectedChannel, setSelectedChannel] = useState('bryanjohnson');
  const [activeTab, setActiveTab] = useState('overview');
  const [searchQuery, setSearchQuery] = useState('');

  const channels = [
    { id: 'bryanjohnson', name: 'Bryan Johnson', videos: 127, insights: 342 },
    { id: 'lexfridman', name: 'Lex Fridman', videos: 89, insights: 567 },
    { id: 'karpathy', name: 'Andrej Karpathy', videos: 45, insights: 234 }
  ];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'insights', label: 'Key Insights', icon: Brain },
    { id: 'knowledge-graph', label: 'Knowledge Graph', icon: Network },
    { id: 'themes', label: 'Themes', icon: Tag },
    { id: 'transcripts', label: 'Transcripts', icon: FileText }
  ];

  const mockInsights = {
    overview: {
      totalVideos: 127,
      totalHours: 89.5,
      keyTopics: ['Longevity', 'Biohacking', 'Health Optimization', 'Sleep', 'Exercise'],
      avgSentiment: 0.72,
      totalInsights: 342
    },
    keyInsights: [
      {
        id: 1,
        title: 'Sleep Quality Optimization',
        description: 'Consistent discussion about sleep tracking, sleep hygiene, and the importance of deep sleep cycles for longevity.',
        confidence: 0.94,
        frequency: 67,
        relatedVideos: 23
      },
      {
        id: 2,
        title: 'Measurement-Driven Approach',
        description: 'Heavy emphasis on biomarker tracking, continuous glucose monitoring, and data-driven health decisions.',
        confidence: 0.91,
        frequency: 89,
        relatedVideos: 34
      },
      {
        id: 3,
        title: 'Anti-Aging Protocols',
        description: 'Detailed coverage of anti-aging supplements, therapies, and lifestyle interventions with scientific backing.',
        confidence: 0.88,
        frequency: 78,
        relatedVideos: 28
      }
    ],
    themes: [
      { name: 'Longevity Research', count: 45, growth: '+12%' },
      { name: 'Sleep Optimization', count: 38, growth: '+8%' },
      { name: 'Biomarker Tracking', count: 34, growth: '+15%' },
      { name: 'Exercise Science', count: 29, growth: '+5%' },
      { name: 'Nutritional Science', count: 26, growth: '+10%' },
      { name: 'Mental Health', count: 22, growth: '+18%' }
    ]
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-600">{mockInsights.overview.totalVideos}</div>
          <div className="text-sm text-gray-600">Total Videos</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-green-600">{mockInsights.overview.totalHours}h</div>
          <div className="text-sm text-gray-600">Content Hours</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-blue-600">{mockInsights.overview.totalInsights}</div>
          <div className="text-sm text-gray-600">Key Insights</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-purple-600">{mockInsights.overview.keyTopics.length}</div>
          <div className="text-sm text-gray-600">Main Topics</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-orange-600">{Math.round(mockInsights.overview.avgSentiment * 100)}%</div>
          <div className="text-sm text-gray-600">Positive Sentiment</div>
        </div>
      </div>

      {/* Key Topics */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Primary Topics</h3>
        <div className="flex flex-wrap gap-2">
          {mockInsights.overview.keyTopics.map((topic, index) => (
            <span
              key={topic}
              className="px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm font-medium"
            >
              {topic}
            </span>
          ))}
        </div>
      </div>

      {/* Channel Summary */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Channel Overview</h3>
        <div className="prose max-w-none">
          <p className="text-gray-600 leading-relaxed">
            Bryan Johnson's content focuses heavily on longevity research and biohacking practices.
            The channel demonstrates a strong emphasis on data-driven health optimization, with recurring
            themes around sleep quality, biomarker tracking, and anti-aging protocols. Content analysis
            reveals a systematic approach to health with heavy reliance on scientific research and
            personal experimentation.
          </p>
        </div>
      </div>
    </div>
  );

  const renderInsights = () => (
    <div className="space-y-6">
      {mockInsights.keyInsights.map((insight) => (
        <motion.div
          key={insight.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="flex items-start justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">{insight.title}</h3>
            <div className="flex items-center space-x-2">
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                insight.confidence > 0.9
                  ? 'bg-green-100 text-green-800'
                  : insight.confidence > 0.8
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                {Math.round(insight.confidence * 100)}% confidence
              </span>
            </div>
          </div>

          <p className="text-gray-600 mb-4">{insight.description}</p>

          <div className="flex items-center space-x-6 text-sm text-gray-500">
            <div className="flex items-center space-x-1">
              <MessageSquare className="w-4 h-4" />
              <span>{insight.frequency} mentions</span>
            </div>
            <div className="flex items-center space-x-1">
              <FileText className="w-4 h-4" />
              <span>{insight.relatedVideos} videos</span>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );

  const renderKnowledgeGraph = () => (
    <div className="card">
      <div className="text-center py-12">
        <Network className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Knowledge Graph Visualization</h3>
        <p className="text-gray-600 mb-6">
          Interactive knowledge graph showing relationships between concepts, people, and topics
        </p>
        <div className="bg-gray-100 rounded-lg p-8 mb-6">
          <div className="text-gray-500">
            [Knowledge Graph Visualization would be rendered here]
            <br />
            This would show an interactive network graph with nodes representing:
            <br />
            • Key concepts and topics
            <br />
            • People mentioned
            <br />
            • Relationships and connections
            <br />
            • Frequency and strength of associations
          </div>
        </div>
        <button className="btn-primary">
          <Download className="w-4 h-4 mr-2" />
          Export Graph Data
        </button>
      </div>
    </div>
  );

  const renderThemes = () => (
    <div className="space-y-4">
      {mockInsights.themes.map((theme, index) => (
        <motion.div
          key={theme.name}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="card"
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900">{theme.name}</h3>
              <p className="text-sm text-gray-600">{theme.count} occurrences</p>
            </div>
            <div className="text-right">
              <div className={`text-sm font-medium ${
                theme.growth.startsWith('+') ? 'text-green-600' : 'text-red-600'
              }`}>
                {theme.growth}
              </div>
              <div className="text-xs text-gray-500">vs. last period</div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );

  const renderTranscripts = () => (
    <div className="card">
      <div className="text-center py-12">
        <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Transcript Search</h3>
        <p className="text-gray-600 mb-6">
          Search through all video transcripts to find specific topics or quotes
        </p>
        <div className="max-w-md mx-auto">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search transcripts..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'insights':
        return renderInsights();
      case 'knowledge-graph':
        return renderKnowledgeGraph();
      case 'themes':
        return renderThemes();
      case 'transcripts':
        return renderTranscripts();
      default:
        return renderOverview();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Channel Insights</h1>

          {/* Channel Selector */}
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
            <div className="flex items-center space-x-4">
              <select
                value={selectedChannel}
                onChange={(e) => setSelectedChannel(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                {channels.map((channel) => (
                  <option key={channel.id} value={channel.id}>
                    {channel.name} ({channel.videos} videos)
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center space-x-2">
              <button className="btn-secondary">
                <Share2 className="w-4 h-4 mr-2" />
                Share
              </button>
              <button className="btn-primary">
                <Download className="w-4 h-4 mr-2" />
                Export
              </button>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-8">
          <nav className="-mb-px flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {renderTabContent()}
        </motion.div>
      </div>
    </div>
  );
};

export default InsightsPage;
