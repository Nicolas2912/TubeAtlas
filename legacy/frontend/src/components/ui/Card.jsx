import { motion } from 'framer-motion';

const Card = ({
  children,
  className = '',
  animate = false,
  hover = false,
  padding = 'default',
  ...props
}) => {
  const baseStyles = 'bg-white rounded-xl border border-gray-200';

  const paddingStyles = {
    none: '',
    sm: 'p-4',
    default: 'p-6',
    lg: 'p-8'
  };

  const hoverStyles = hover ? 'hover:shadow-lg hover:-translate-y-1 cursor-pointer' : 'shadow-lg';

  const cardClasses = `
    ${baseStyles}
    ${paddingStyles[padding]}
    ${hoverStyles}
    transition-all duration-200
    ${className}
  `.trim();

  if (animate) {
    return (
      <motion.div
        className={cardClasses}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        {...props}
      >
        {children}
      </motion.div>
    );
  }

  return (
    <div className={cardClasses} {...props}>
      {children}
    </div>
  );
};

export default Card;
