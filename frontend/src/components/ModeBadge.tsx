/**
 * ModeBadge Component
 * 
 * Laboratuvar ve görselleştirmelerde işlem modunu gösteren badge.
 * GERÇEK: Gerçek hesaplama ve model inference
 * SİMÜLASYON: Eğitim amaçlı simülasyon
 * DEMO: Örnek veri ile demo
 */

import React from 'react';
import { Activity, Cpu, Sparkles } from 'lucide-react';

export type OperationMode = 'real' | 'simulation' | 'demo' | 'pending';

interface ModeBadgeProps {
  mode: OperationMode;
  className?: string;
  showIcon?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const modeConfig = {
  real: {
    label: 'GERÇEK',
    description: 'Gerçek hesaplama ve model inference',
    bgColor: 'bg-green-100 dark:bg-green-900/30',
    textColor: 'text-green-800 dark:text-green-300',
    borderColor: 'border-green-300 dark:border-green-700',
    icon: Activity,
  },
  simulation: {
    label: 'SİMÜLASYON',
    description: 'Eğitim amaçlı matematiksel simülasyon',
    bgColor: 'bg-blue-100 dark:bg-blue-900/30',
    textColor: 'text-blue-800 dark:text-blue-300',
    borderColor: 'border-blue-300 dark:border-blue-700',
    icon: Cpu,
  },
  demo: {
    label: 'DEMO',
    description: 'Örnek veri ile arayüz tanıtımı',
    bgColor: 'bg-purple-100 dark:bg-purple-900/30',
    textColor: 'text-purple-800 dark:text-purple-300',
    borderColor: 'border-purple-300 dark:border-purple-700',
    icon: Sparkles,
  },
  pending: {
    label: 'BEKLENİYOR',
    description: 'Veri henüz yüklenmedi',
    bgColor: 'bg-gray-100 dark:bg-gray-800',
    textColor: 'text-gray-600 dark:text-gray-400',
    borderColor: 'border-gray-300 dark:border-gray-600',
    icon: Activity,
  },
};

const sizeClasses = {
  sm: 'text-[10px] px-1.5 py-0.5',
  md: 'text-xs px-2 py-1',
  lg: 'text-sm px-3 py-1.5',
};

export default function ModeBadge({ 
  mode, 
  className = '', 
  showIcon = true,
  size = 'md'
}: ModeBadgeProps) {
  const config = modeConfig[mode];
  const Icon = config.icon;

  return (
    <div
      className={`
        inline-flex items-center gap-1 
        rounded-full border font-semibold uppercase tracking-wide
        ${config.bgColor} ${config.textColor} ${config.borderColor}
        ${sizeClasses[size]}
        ${className}
      `}
      title={config.description}
    >
      {showIcon && <Icon className={size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-4 h-4' : 'w-3.5 h-3.5'} />}
      <span>{config.label}</span>
    </div>
  );
}

/**
 * ModeIndicator Component
 * 
 * Daha detaylı mod göstergesi - açıklama ile
 */

interface ModeIndicatorProps {
  mode: OperationMode;
  title?: string;
  description?: string;
  className?: string;
}

export function ModeIndicator({ 
  mode, 
  title, 
  description,
  className = '' 
}: ModeIndicatorProps) {
  const config = modeConfig[mode];
  const Icon = config.icon;

  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg border ${config.borderColor} ${config.bgColor} ${className}`}>
      <div className={`p-2 rounded-lg ${config.bgColor} ${config.textColor}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h4 className={`font-semibold ${config.textColor}`}>
            {title || config.label}
          </h4>
        </div>
        <p className={`text-sm ${config.textColor} opacity-90`}>
          {description || config.description}
        </p>
      </div>
    </div>
  );
}

/**
 * Hook to determine mode from data
 */
export function useOperationMode(data?: { mode?: string; stats?: { mode?: string } }): OperationMode {
  if (!data) return 'pending';
  
  const mode = data.mode || data.stats?.mode;
  
  if (mode === 'real' || mode === 'REAL') return 'real';
  if (mode === 'simulation' || mode === 'SIMULATION') return 'simulation';
  if (mode === 'demo' || mode === 'DEMO') return 'demo';
  
  return 'pending';
}
