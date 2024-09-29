translations = {
    'average_reward': {
        'zh': '平均奖励',
        'en': 'Average Reward',
        'es': 'Recompensa promedio',
    },
    'reward_std': {
        'zh': '奖励标准差',
        'en': 'Reward Standard Deviation',
        'es': 'Desviación estándar de la recompensa',
    },
    'max_reward': {
        'zh': '最高奖励',
        'en': 'Maximum Reward',
        'es': 'Recompensa máxima',
    },
    'min_reward': {
        'zh': '最低奖励',
        'en': 'Minimum Reward',
        'es': 'Recompensa mínima',
    },
    'average_episode_length': {
        'zh': '平均回合长度',
        'en': 'Average Episode Length',
        'es': 'Longitud promedio del episodio',
    },
    'success_rate': {
        'zh': '成功率',
        'en': 'Success Rate',
        'es': 'Tasa de éxito',
    },
    'test_episodes': {
        'zh': '测试回合数',
        'en': 'Test Episodes',
        'es': 'Episodios de prueba',
    },
    'test_duration': {
        'zh': '测试用时(秒)',
        'en': 'Test Duration (s)',
        'es': 'Duración de la prueba (s)',
    },
    'total_reward': {
        'zh': '总奖励',
        'en': 'Total Reward',
        'es': 'Recompensa total',
    },
    'maximum_episodes_reached': {
        'zh': '达到最大回合数',
        'en': 'Maximum episodes reached',
        'es': 'Se alcanzó el número máximo de episodios',
    },
    'maximum_total_steps_reached': {
        'zh': '达到最大总步数',
        'en': 'Maximum total steps reached',
        'es': 'Se alcanzó el número máximo de pasos totales',
    },
    'maximum_runtime_reached': {
        'zh': '达到最大运行时间',
        'en': 'Maximum runtime reached',
        'es': 'Se alcanzó el tiempo máximo de ejecución',
    },
    'reward_threshold_reached': {
        'zh': '达到奖励阈值',
        'en': 'Reward threshold reached',
        'es': 'Se alcanzó el umbral de recompensa',
    },
    'exceeded_maximum_reward_threshold': {
        'zh': '超过最大奖励阈值',
        'en': 'Exceeded maximum reward threshold',
        'es': 'Se superó el umbral máximo de recompensa',
    },
    'no_performance_improvement_absolute': {
        'zh': '模型性能不再改善（绝对值）',
        'en': 'No performance improvement (absolute)',
        'es': 'Sin mejora de rendimiento (absoluto)',
    },
    'no_performance_improvement_ratio': {
        'zh': '模型性能不再改善（比例）',
        'en': 'No performance improvement (ratio)',
        'es': 'Sin mejora de rendimiento (ratio)',
    },
    'checkpoint_saved': {
        'zh': '检查点已保存',
        'en': 'Checkpoint saved',
        'es': 'Punto de control guardado',
    },
    'final_model_saved': {
        'zh': '最终模型已保存',
        'en': 'Final model saved',
        'es': 'Modelo final guardado',
    },
    'final_model_not_saved': {
        'zh': '最终模型未保存，因为未指定保存路径',
        'en': 'Final model not saved because no path was specified',
        'es': 'El modelo final no se guardó porque no se especificó una ruta',
    },
}

def get_text(key, lang='en'):
    return translations.get(key, {}).get(lang, key)