#!/bin/bash

# Parking Detector v2.0 - Скрипт переключения устройства
# Автор: Parking Detector Team
# Версия: 2.0

echo "🔧 Parking Detector v2.0 - Переключение устройства"
echo "=================================================="

# Проверка аргументов
if [ $# -eq 0 ]; then
    echo "Использование: $0 [cpu|cuda|auto]"
    echo ""
    echo "Режимы:"
    echo "  cpu  - Принудительное использование CPU"
    echo "  cuda - Принудительное использование CUDA"
    echo "  auto - Автоматическое определение (по умолчанию)"
    exit 1
fi

MODE=$1
CONFIG_FILE="../config/settings.json"

# Проверка существования конфигурационного файла
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Конфигурационный файл не найден: $CONFIG_FILE"
    exit 1
fi

# Создание резервной копии
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
echo "💾 Создана резервная копия: ${CONFIG_FILE}.backup"

case $MODE in
    "cpu")
        echo "🖥️ Переключение на CPU режим..."
        # Обновление конфигурации для CPU
        python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['force_cpu'] = True
config['device'] = 'cpu'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('✅ Конфигурация обновлена для CPU режима')
"
        ;;
    "cuda")
        echo "🎮 Переключение на CUDA режим..."
        # Обновление конфигурации для CUDA
        python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['force_cpu'] = False
config['device'] = 'cuda'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('✅ Конфигурация обновлена для CUDA режима')
"
        ;;
    "auto")
        echo "🤖 Переключение на автоматический режим..."
        # Обновление конфигурации для автоматического режима
        python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
config['force_cpu'] = False
config['device'] = 'auto'
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('✅ Конфигурация обновлена для автоматического режима')
"
        ;;
    *)
        echo "❌ Неизвестный режим: $MODE"
        echo "Доступные режимы: cpu, cuda, auto"
        exit 1
        ;;
esac

echo ""
echo "🎉 Переключение завершено!"
echo "📋 Текущая конфигурация:"
cat "$CONFIG_FILE" | grep -E "(force_cpu|device)"
echo ""
echo "🚀 Запустите программу: cd ../src && python3 parking_detector.py"
