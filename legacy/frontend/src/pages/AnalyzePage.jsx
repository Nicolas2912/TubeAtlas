import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Youtube,
  Search,
  Play,
  Download,
  Brain,
  Network,
  AlertCircle,
  CheckCircle,
  Loader,
  ExternalLink
} from 'lucide-react';

const AnalyzePage = () => {
  const [channelUrl, setChannelUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState('');

  const steps = [
    {
      title: 'Extract Transcripts',
      description: 'Downloading video transcripts and metadata',
      icon: Download
    },
    {
      title: 'Process Content',
      description: 'Analyzing content with AI models',
      icon: Brain
    },
    {
      title: 'Build Knowledge Graph',
      description: 'Creating relationships and insights',
      icon: Network
    },
    {
      title: 'Generate Insights',
      description: 'Extracting key themes and patterns',
      icon: CheckCircle
    }
  ];

  const exampleChannels = [
    { name: 'Andrej Karpathy', url: '@karpathy', description: 'AI Education & Research' },
    { name: 'Lex Fridman', url: '@lexfridman', description: 'AI & Technology Interviews' },
    { name: 'Two Minute Papers', url: '@TwoMinutePapers', description: 'AI Research Summaries' },
    { name: 'Bryan Johnson', url: '@bryanjohnsonlive', description: 'Longevity & Biohacking' }
  ];

  const handleAnalyze = async () => {
    if (!channelUrl.trim()) {
      setError('Please enter a valid YouTube channel URL or handle');
      return;
    }

    setError('');
    setIsAnalyzing(true);
    setCurrentStep(0);

    // Simulate analysis process
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
    }

    setIsAnalyzing(false);
    // Here you would typically redirect to results or update UI with completion status
  };

  const validateChannelUrl = (url) => {
    const patterns = [
      /^@[\w\d]+$/,  // @username format
      /^https?:\/\/(www\.)?youtube\.com\/channel\/[\w-]+/,  // Channel URL
      /^https?:\/\/(www\.)?youtube\.com\/c\/[\w-]+/,  // Custom URL
      /^https?:\/\/(www\.)?youtube\.com\/@[\w\d]+/  // Handle URL
    ];

    return patterns.some(pattern => pattern.test(url.trim()));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Analyze YouTube Channel
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Enter a YouTube channel URL or handle to extract transcripts, build knowledge graphs,
            and generate deep insights from the content.
          </p>
        </motion.div>

        {/* Analysis Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="card mb-8"
        >
          <div className="mb-6">
            <label htmlFor="channel-url" className="block text-sm font-medium text-gray-700 mb-2">
              YouTube Channel URL or Handle
            </label>
            <div className="relative">
              <Youtube className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                id="channel-url"
                type="text"
                value={channelUrl}
                onChange={(e) => setChannelUrl(e.target.value)}
                placeholder="@username or https://youtube.com/channel/..."
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
                disabled={isAnalyzing}
              />
            </div>
            {error && (
              <div className="flex items-center space-x-2 mt-2 text-red-600">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !validateChannelUrl(channelUrl)}
            className={`w-full flex items-center justify-center space-x-2 py-3 px-6 rounded-lg font-medium transition-all duration-200 ${
              isAnalyzing || !validateChannelUrl(channelUrl)
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-primary-600 hover:bg-primary-700 text-white transform hover:scale-105'
            }`}
          >
            {isAnalyzing ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                <span>Start Analysis</span>
              </>
            )}
          </button>
        </motion.div>

        {/* Analysis Progress */}
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            transition={{ duration: 0.5 }}
            className="card mb-8"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-6">Analysis Progress</h3>
            <div className="space-y-4">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = index === currentStep;
                const isCompleted = index < currentStep;

                return (
                  <div key={step.title} className="flex items-center space-x-4">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      isCompleted
                        ? 'bg-green-100 text-green-600'
                        : isActive
                        ? 'bg-primary-100 text-primary-600'
                        : 'bg-gray-100 text-gray-400'
                    }`}>
                      {isCompleted ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : isActive ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : (
                        <Icon className="w-5 h-5" />
                      )}
                    </div>
                    <div className="flex-1">
                      <h4 className={`font-medium ${
                        isCompleted ? 'text-green-900' : isActive ? 'text-primary-900' : 'text-gray-500'
                      }`}>
                        {step.title}
                      </h4>
                      <p className={`text-sm ${
                        isCompleted ? 'text-green-600' : isActive ? 'text-primary-600' : 'text-gray-400'
                      }`}>
                        {step.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}

        {/* Example Channels */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Try with Popular Channels
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {exampleChannels.map((channel) => (
              <button
                key={channel.name}
                onClick={() => setChannelUrl(channel.url)}
                disabled={isAnalyzing}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-all duration-200 text-left group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div>
                  <h4 className="font-medium text-gray-900 group-hover:text-primary-900">
                    {channel.name}
                  </h4>
                  <p className="text-sm text-gray-600">{channel.description}</p>
                  <p className="text-xs text-gray-400 mt-1">{channel.url}</p>
                </div>
                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-primary-600" />
              </button>
            ))}
          </div>
        </motion.div>

        {/* Info Box */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6"
        >
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-blue-900 mb-2">Analysis Information</h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• Analysis time depends on the number of videos and their length</li>
                <li>• Large channels may take several minutes to process completely</li>
                <li>• You'll be able to view results and download data once complete</li>
                <li>• All processing is done securely and data is not stored permanently</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default AnalyzePage;
