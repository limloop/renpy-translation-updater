import os
import re
import time
import json
import fcntl
import hashlib
import threading
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from yandexfreetranslate import YandexFreeTranslate
from concurrent.futures import ThreadPoolExecutor, as_completed

class TranslationCache:
    """Потокобезопасный кеш переводов с корректным управлением файлами"""
    def __init__(self, cache_file="translation_cache.jsonl"):
        self.cache_file = cache_file
        self.cache = {}
        self._lock = threading.Lock()  # Для синхронизации операций
        self._load_cache()

    def _load_cache(self):
        """Загружает кеш из файла с обработкой ошибок"""
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                # Блокировка на время чтения
                self._acquire_lock(f)
                try:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            self.cache[entry['hash']] = entry['translation']
                        except json.JSONDecodeError:
                            continue
                finally:
                    self._release_lock(f)
        except IOError as e:
            print(f"Ошибка чтения кеша: {e}")

    def _acquire_lock(self, file_obj):
        """Блокировка файла"""
        if os.name == 'posix':
            fcntl.flock(file_obj, fcntl.LOCK_SH)
        # Для Windows можно использовать msvcrt.locking

    def _release_lock(self, file_obj):
        """Разблокировка файла"""
        if os.name == 'posix':
            fcntl.flock(file_obj, fcntl.LOCK_UN)

    def get_hash(self, text: str) -> str:
        """Генерирует MD5 хеш текста"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_cached_translation(self, text: str, dest_lang: str) -> Optional[str]:
        """Получает перевод из кеша"""
        text_hash = self.get_hash(f"{dest_lang}_{text}")
        with self._lock:
            return self.cache.get(text_hash)

    def add_translation(self, text: str, translation: str, dest_lang: str) -> bool:
        """Добавляет перевод в кеш (атомарная операция)"""
        text_hash = self.get_hash(f"{dest_lang}_{text}")
        
        with self._lock:
            if text_hash in self.cache:
                return False

            entry = {
                'hash': text_hash,
                'original': text,
                'translation': translation
            }

            try:
                # Открываем файл в режиме добавления
                with open(self.cache_file, 'a', encoding='utf-8') as f:
                    self._acquire_lock(f)
                    try:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        self.cache[text_hash] = translation
                        return True
                    finally:
                        self._release_lock(f)
            except IOError as e:
                print(f"Ошибка записи в кеш: {e}")
                return False

    def compact_cache(self):
        """Оптимизирует файл кеша (выполнять периодически)"""
        with self._lock:
            if not os.path.exists(self.cache_file):
                return

            temp_file = self.cache_file + '.tmp'
            try:
                with open(temp_file, 'w', encoding='utf-8') as tmp_f, \
                     open(self.cache_file, 'r', encoding='utf-8') as src_f:
                    
                    self._acquire_lock(src_f)
                    try:
                        # Записываем только уникальные записи
                        seen_hashes = set()
                        for line in src_f:
                            try:
                                entry = json.loads(line.strip())
                                if entry['hash'] not in seen_hashes:
                                    tmp_f.write(line)
                                    seen_hashes.add(entry['hash'])
                            except json.JSONDecodeError:
                                continue
                    finally:
                        self._release_lock(src_f)

                # Атомарная замена файла
                os.replace(temp_file, self.cache_file)
            except Exception as e:
                print(f"Ошибка оптимизации кеша: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)


def parse_simple_line(line: str) -> Dict:
    """Парсит строку формата old "..." или new "..." """
    if '"' not in line:
        return None

    start_text = line.find('"')
    end_text = line.rfind('"')
    
    entry_type = line[:start_text].strip()
    text = line[start_text+1:end_text].strip()
    
    return {
        'entry_type': 'simple',
        'type': entry_type,
        'text': text
    }

def parse_named_pair(comment_line: str, content_line: str) -> Dict:
    """Парсит именованную пару (#name / name)"""
    # Обработка комментария
    start_text = comment_line.find('"')
    end_text = comment_line.rfind('"')

    name = comment_line[1:start_text].strip()
    comment_text = comment_line[start_text+1: end_text]

    # Обработка содержимого
    start_text = content_line.find('"')
    end_text = content_line.rfind('"')
    
    content_name = content_line[:start_text].strip()
    content_text = content_line[start_text+1: end_text]

    if name != content_name:
        return None

    return {
        'entry_type': 'named',
        'name': name,
        'old': comment_text,
        'new': content_text
    }

def parse_quoted_pair(comment_line: str, content_line: str) -> Dict:
    """Парсит пару с кавычками (#"..." / "...")"""
    # Обработка комментария
    start_text = comment_line.find('"')
    end_text = comment_line.rfind('"')
    
    comment_text = comment_line[start_text+1:end_text]

    # Обработка содержимого
    start_text = content_line.find('"')
    end_text = content_line.rfind('"')
    
    content_text = content_line[start_text+1:end_text]

    return {
        'entry_type': 'quoted',
        'old': comment_text,
        'new': content_text
    }

def normalize_blocks(blocks: List[Dict]) -> List[Dict]:
    """Приводит все блоки к единому формату"""
    normalized = []
    
    for block in blocks:
        if block['type'] == 'strings':
            # Для блока strings группируем old/new пары
            entries = []
            i = 0
            while i < len(block['entries']):
                if (i+1 < len(block['entries']) and 
                    block['entries'][i]['entry_type'] == 'simple' and
                    block['entries'][i]['type'] == 'old' and
                    block['entries'][i+1]['entry_type'] == 'simple' and
                    block['entries'][i+1]['type'] == 'new'):
                    
                    entries.append({
                        'old': block['entries'][i]['text'],
                        'new': block['entries'][i+1]['text']
                    })
                    i += 2
                else:
                    i += 1
            
            if entries:
                normalized.append({
                    'type': 'strings',
                    'language': block['language'],
                    'entries': entries
                })
        else:
            # Для других блоков сохраняем как есть
            normalized_block = {
                'type': block['type'],
                'language': block['language'],
                'entries': []
            }
            
            for entry in block['entries']:
                if entry['entry_type'] == 'named':
                    normalized_block['entries'].append({
                        'type': 'named',
                        'name': entry['name'],
                        'old': entry['old'],
                        'new': entry['new']
                    })
                elif entry['entry_type'] == 'quoted':
                    normalized_block['entries'].append({
                        'type': 'quoted',
                        'old': entry['old'],
                        'new': entry['new']
                    })
                elif entry['entry_type'] == 'simple' and entry['type'] == 'old':
                    # Одиночные old без new (редкий случай)
                    normalized_block['entries'].append({
                        'type': 'simple',
                        'old': entry['text'],
                        'new': ''
                    })

            if normalized_block['entries']:
                normalized.append(normalized_block)

    return normalized

def parse_translation_file(file_path: str) -> List[Dict]:
    """Парсит файл перевода без использования регулярных выражений"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    current_block = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Обработка начала блока translate
        if line.startswith('translate'):
            if current_block is not None:
                blocks.append(current_block)
            
            parts = line.split()
            if len(parts) >= 3:
                current_block = {
                    'type': parts[2].rstrip(':'),
                    'language': parts[1],
                    'entries': []
                }
            i += 1
            continue

        # Обработка содержимого блока
        if current_block is not None:
            # Стандартные пары old/new
            if line.startswith('old "') or line.startswith('new "'):
                entry = parse_simple_line(line)
                if entry:
                    current_block['entries'].append(entry)
                i += 1
                continue

            # Пары с кавычками (#"..." / "...")
            elif line.startswith('# "') and i+1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line.startswith('"'):
                    entry = parse_quoted_pair(line, next_line)
                    if entry:
                        current_block['entries'].append(entry)
                        i += 2
                        continue

            # Именованные пары (#name / name)
            elif line.startswith('# ') and i+1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line.split('"')[0].strip().isidentifier():
                    entry = parse_named_pair(line, next_line)
                    if entry:
                        current_block['entries'].append(entry)
                        i += 2
                        continue

        i += 1

    # Добавляем последний блок
    if current_block is not None:
        blocks.append(current_block)

    return normalize_blocks(blocks)

def merge_translations(old_blocks: List[Dict], new_blocks: List[Dict]) -> List[Dict]:
    """
    Объединяет старые и новые переводы, сохраняя структуру блоков.
    Переносит существующие переводы из old_blocks в new_blocks, где new пусто.
    """
    # Создаем словари для быстрого поиска старых переводов
    old_translations = {
        'strings': {},    # {old_text: new_translation}
        'named': {},      # {(name, old_text): new_translation}
        'quoted': {}      # {comment: new_translation}
    }

    # Сначала собираем все старые переводы
    for block in old_blocks:
        if block['type'] == 'strings':
            for entry in block['entries']:
                old_translations['strings'][entry['old']] = entry['new']
        elif block['type'] == 'named':
            key = (block['name'], block['old'])
            old_translations['named'][key] = block['new']
        elif block['type'] == 'quoted':
            old_translations['quoted'][block['old']] = block['new']

    merged_blocks = []
    
    for new_block in new_blocks:
        merged_block = new_block.copy()
        
        if new_block['type'] == 'strings':
            # Обрабатываем блок strings
            merged_entries = []
            for entry in new_block['entries']:
                # Берем старый перевод, если новый пустой
                if not entry['new'] and entry['old'] in old_translations['strings']:
                    merged_entries.append({
                        'old': entry['old'],
                        'new': old_translations['strings'][entry['old']]
                    })
                else:
                    merged_entries.append(entry)
            
            merged_block['entries'] = merged_entries
        
        elif new_block['type'] == 'named':
            # Обрабатываем именованные блоки
            key = (new_block['name'], new_block['old'])
            if key in old_translations['named'] and not new_block['new']:
                merged_block['new'] = old_translations['named'][key]
        
        elif new_block['type'] == 'quoted':
            # Обрабатываем блоки с кавычками
            if new_block['old'] in old_translations['quoted'] and not new_block['new']:
                merged_block['new'] = old_translations['quoted'][new_block['old']]
        
        merged_blocks.append(merged_block)
    
    return merged_blocks

def contains_special_chars(text: str) -> bool:
    """Проверяет, содержит ли текст специальные символы"""
    special_chars = r'[{\[\]\/<>@#$%^&*_=+\\|~`]'
    return bool(re.search(special_chars, text))

def translate_missing_blocks(trans_file, blocks: List[Dict], src_lang='en', dest_lang='ru') -> Tuple[List[Dict], int, int]:
    """
    Переводит отсутствующие переводы с кешированием результатов.
    Возвращает:
    - обновленные блоки с переводами
    - количество переведенных строк
    - количество строк из кеша
    """
    translator = YandexFreeTranslate(api = "ios")
    cache = TranslationCache()
    
    translated_count = 0
    total_to_translate = 0
    
    # Подсчет строк для перевода
    for block in blocks:
        for entry in block['entries']:
            if not entry['new'] and not contains_special_chars(entry['old']):
                total_to_translate += 1
    
    if total_to_translate == 0:
        return blocks, 0
    
    # Создаем копию блоков для модификации
    translated_blocks = [block.copy() for block in blocks]
    
    with tqdm(total=total_to_translate, desc=trans_file[:min(len(trans_file), 20)]) as pbar:
        for block_idx, block in enumerate(blocks):
            for entry_idx, entry in enumerate(block['entries']):
                if not entry['new'] and not contains_special_chars(entry['old']):
                    # Проверяем кеш
                    cached = cache.get_cached_translation(entry['old'], dest_lang)
                    if cached:
                        translated_blocks[block_idx]['entries'][entry_idx]['new'] = cached.replace('"', '\\"')
                        pbar.update(1)
                        continue
                    
                    # Пытаемся перевести
                    success = False
                    for attempt in range(3):
                        try:
                            translation = translator.translate(src_lang, dest_lang, entry['old'])
                            
                            # Сохраняем в кеш
                            cache.add_translation(entry['old'], translation, dest_lang)
                            
                            translated_blocks[block_idx]['entries'][entry_idx]['new'] = translation.replace('"', '\\"')
                            translated_count += 1
                            success = True
                            break
                        except Exception as e:
                            if attempt == 2:
                                print(f"\nОшибка перевода строки: '{entry['old']}' - {str(e)}")
                            time.sleep(1 + attempt)
                    
                    pbar.update(1)
                    if not success:
                        translated_blocks[block_idx]['entries'][entry_idx]['new'] = entry['old'].replace('"', '\\"')
    
    return translated_blocks, translated_count

def generate_translation_file(blocks: List[Dict], output_path: str):
    """Генерирует файл перевода с правильной группировкой всех типов блоков"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Разделяем блоки на strings и другие
        strings_blocks = [b for b in blocks if b['type'] == 'strings']
        other_blocks = [b for b in blocks if b['type'] != 'strings']
        
        # Записываем strings блоки
        if strings_blocks:
            language = strings_blocks[0]['language']
            f.write(f"translate {language} strings:\n")
            
            for block in strings_blocks:
                for entry in block['entries']:
                    f.write(f"    old \"{entry['old']}\"\n")
                    f.write(f"    new \"{entry['new'] if entry['new'] else entry['old']}\"\n")
        
        # Записываем остальные блоки
        for block in other_blocks:
            f.write(f"translate {block['language']} {block['type']}:\n")
            
            for entry in block['entries']:
                if entry['type'] == 'named':
                    f.write(f"    # {entry['name']} \"{entry['old']}\"\n")
                    f.write(f"    {entry['name']} \"{entry['new'] if entry['new'] else entry['old']}\"\n\n")
                elif entry['type'] == 'quoted':
                    f.write(f"    # \"{entry['old']}\"\n")
                    f.write(f"    \"{entry['new'] if entry['new'] else entry['old']}\"\n\n")
                elif entry['type'] == 'simple':
                    f.write(f"    old \"{entry['old']}\"\n")
                    f.write(f"    new \"{entry['new'] if entry['new'] else entry['old']}\"\n\n")
            
            f.write("\n")

def process_file(filename, old_dir, new_dir, output_dir, src_lang='en', dest_lang='ru'):
    """Обрабатывает один файл перевода"""
    if not filename.endswith('.rpy'):
        return None
    
    old_path = os.path.join(old_dir, filename)
    new_path = os.path.join(new_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    # Парсим файлы
    old_blocks = parse_translation_file(old_path) if os.path.exists(old_path) else []
    new_blocks = parse_translation_file(new_path)

    # Объединяем переводы
    merged_blocks = merge_translations(old_blocks, new_blocks) if old_blocks else new_blocks
    
    # Переводим отсутствующие строки
    translated_blocks, translated_count = translate_missing_blocks(filename, merged_blocks, src_lang, dest_lang)
    
    # Сохраняем результат
    generate_translation_file(translated_blocks, output_path)
    
    return filename, translated_count

def process_all_files(old_dir, new_dir, output_dir, src_lang='en', dest_lang='ru', max_workers=8):
    """Обрабатывает все файлы в многопоточном режиме"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files_to_process = [f for f in os.listdir(new_dir) if f.endswith('.rpy')]
    total_files = len(files_to_process)
    
    if not files_to_process:
        print("Нет .rpy файлов для обработки")
        return
    
    print(f"Начата обработка {total_files} файлов в {max_workers} потоках...")
    
    total_translated = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, old_dir, new_dir, output_dir, src_lang, dest_lang): f 
            for f in files_to_process
        }
        
        with tqdm(total=total_files, desc="Обработка файлов") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    filename, translated_count = result
                    total_translated += translated_count
                    pbar.set_postfix_str(f"Переведено: {total_translated} строк")
                pbar.update(1)
    
    print(f"\nОбработка завершена.")

def main():
    # Настройки путей
    old_dir = input("Папка tl/* старого перевода")
    new_dir = input("Папка tl/* базового перевода (полученая через renpy)")
    output_dir = input("Папка вывода нового перевода (дополнительные файлы нужно переносить в ручную)")
    
    src_lang = input("Исходный язык (ru/en/...)")
    dest_lang = input("Нужный язык (ru/en/...)")

    process_all_files(old_dir, new_dir, output_dir, max_workers=8)

if __name__ == "__main__":
    main()