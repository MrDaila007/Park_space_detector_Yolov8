#!/bin/bash

# Parking Detector v2.0 - Скрипт установки
# Автор: Parking Detector Team
# Версия: 2.0

echo "🚗 Parking Detector v2.0 - Установка"
echo "======================================"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.8+ и повторите попытку."
    exit 1
fi

# Проверка версии Python
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Требуется Python 3.8+, найден $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION найден"

# Проверка pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 не найден. Установите pip и повторите попытку."
    exit 1
fi

echo "✅ pip3 найден"

# Создание виртуального окружения (опционально)
read -p "🤔 Создать виртуальное окружение? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Виртуальное окружение создано и активировано"
fi

# Установка зависимостей
echo "📦 Установка зависимостей..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Зависимости установлены успешно"
else
    echo "❌ Ошибка при установке зависимостей"
    exit 1
fi

# Создание необходимых директорий
echo "📁 Создание директорий..."
mkdir -p config data models docs examples scripts

# Создание конфигурационного файла по умолчанию
if [ ! -f "config/settings.json" ]; then
    echo "⚙️ Создание конфигурационного файла..."
    cat > config/settings.json << EOF
{
  "universal_detection": true,
  "occupancy_threshold": 0.6,
  "uncertainty_threshold": 0.3,
  "uncertainty_time_threshold": 3.0,
  "frequent_detection_threshold": 10,
  "frequent_detection_window": 10.0,
  "tracked_objects": [2, 67]
}
EOF
    echo "✅ Конфигурационный файл создан"
fi

# Создание файла парковочных мест
if [ ! -f "data/parking_spaces.json" ]; then
    echo "🅿️ Создание файла парковочных мест..."
    echo "[]" > data/parking_spaces.json
    echo "✅ Файл парковочных мест создан"
fi

# Проверка CUDA (опционально)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU обнаружен:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    echo "✅ CUDA ускорение будет доступно"
    echo "💡 Для принудительного использования CPU установите force_cpu=true в config/settings.json"
else
    echo "⚠️ NVIDIA GPU не обнаружен. Программа будет работать на CPU"
    echo "💡 Для принудительного использования CPU установите force_cpu=true в config/settings.json"
fi

echo ""
echo "🎉 Установка завершена успешно!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Настройте видеопоток в файле src/parking_detector.py"
echo "2. Запустите программу: cd src && python3 parking_detector.py"
echo "3. Создайте парковочные места, кликая по углам"
echo "4. Нажмите S для сохранения"
echo ""
echo "📚 Документация: docs/README_Detailed.md"
echo "🔧 Конфигурация: config/settings.json"
echo ""
echo "Удачного использования! 🚗"
