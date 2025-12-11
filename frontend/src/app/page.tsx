import Link from 'next/link'

export default function Home() {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <section className="text-center py-12">
        <h1 className="text-4xl font-bold text-slate-800 dark:text-white mb-4">
          5D Neural Network Regressor
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
          A machine learning system for interpolating 5-dimensional datasets 
          using configurable neural networks.
        </p>
      </section>

      {/* Feature Cards */}
      <section className="grid md:grid-cols-3 gap-6">
        <FeatureCard
          href="/upload"
          title="Upload Dataset"
          description="Upload your 5D dataset in .pkl format for training"
          icon="📁"
          color="blue"
        />
        <FeatureCard
          href="/train"
          title="Train Model"
          description="Configure hyperparameters and train your neural network"
          icon="🧠"
          color="green"
        />
        <FeatureCard
          href="/predict"
          title="Make Predictions"
          description="Input 5 feature values and get instant predictions"
          icon="🎯"
          color="purple"
        />
      </section>

      {/* Info Section */}
      <section className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700">
        <h2 className="text-xl font-semibold text-slate-800 dark:text-white mb-4">
          How It Works
        </h2>
        <ol className="space-y-3 text-slate-600 dark:text-slate-300">
          <li className="flex items-start gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 flex items-center justify-center text-sm font-medium">1</span>
            <span><strong>Upload</strong> your dataset containing X (5 features) and y (target) arrays</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-300 flex items-center justify-center text-sm font-medium">2</span>
            <span><strong>Train</strong> a neural network with customizable architecture and hyperparameters</span>
          </li>
          <li className="flex items-start gap-3">
            <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-300 flex items-center justify-center text-sm font-medium">3</span>
            <span><strong>Predict</strong> output values for new 5D input vectors</span>
          </li>
        </ol>
      </section>
    </div>
  )
}

function FeatureCard({ 
  href, 
  title, 
  description, 
  icon, 
  color 
}: { 
  href: string
  title: string
  description: string
  icon: string
  color: 'blue' | 'green' | 'purple'
}) {
  const colorClasses = {
    blue: 'hover:border-blue-300 dark:hover:border-blue-600',
    green: 'hover:border-green-300 dark:hover:border-green-600',
    purple: 'hover:border-purple-300 dark:hover:border-purple-600',
  }

  return (
    <Link href={href}>
      <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 transition-all duration-200 hover:shadow-md ${colorClasses[color]} cursor-pointer h-full`}>
        <div className="text-4xl mb-4">{icon}</div>
        <h3 className="text-lg font-semibold text-slate-800 dark:text-white mb-2">
          {title}
        </h3>
        <p className="text-slate-600 dark:text-slate-400 text-sm">
          {description}
        </p>
      </div>
    </Link>
  )
}