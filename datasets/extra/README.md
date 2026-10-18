# Дополнительные данные ForteBank: план 40 роликов и проверка источников

## Текущее состояние

Поиск расширен до **45 записей источников и публикаций**: исследовательские наборы,
банковские демонстрации, учебные записи, CCTV поставщиков, государственный видеоархив
и стоки. Проверены доступные страницы на русском, английском, китайском и корейском;
это журнал выполненного поиска, а не утверждение о проверке всего интернета.

Подготовлены **22 локальных видео**: 9 UCF-Crime, 1 INRIA CASSIOPEE, 3 iLinkPro,
1 банковская запись Instablog9ja и 8 DVIDS. В этом обновлении добавлено 13 видео.
Все фрагменты — MOV/H.264, 30 fps, длительностью 15–41 с; всего 548 с / 16440 кадров.
Новые фрагменты полностью декодированы ffprobe, проверены длительности, шаг PTS и SHA-256.
Ранее выполненные проверки 9 UCF-фрагментов сохранены; их хэши повторно сверены.

В `external_candidates.csv` есть две **внешние аннотации с ограниченной областью**:

| Файл | Наблюдение | Аннотация | Ограничение |
|---|---|---|---|
| CAND-INRIA-001.mov | Постановочный посторонний проходит за стойку | V01 analogue, violation; оценка начала 5.75 с, интервал 5.50–6.125 с | Две камеры в одном кадре; исходник 8 fps с надписями; лицензия не установлена |
| CAND-DVIDS-113979.mov | Непрерывный видимый пересчёт над столом | clean только по видимости V08; исходник 52.4–67.4 с | Камера сбоку на уровне человека; нет парного нарушения |

`label_scope` объясняет границы этих аннотаций. Они не означают проверку всех процедур
банка или нарушение правил учреждения, изображённого на записи. `training_eligible=0`
означает, что файл не принят в целевые 40 и не должен автоматически попадать в обучение
или оценку сервиса. У остальных 20 файлов `label/t_sec/type_id` пусты.
`labels.csv` остаётся честным шаблоном из 40 запланированных EXT, без выдуманных t_sec.

Повторный просмотр исключил ложные кандидаты: UCF Normal056 показывает **проводную
трубку**, а в DVIDS804414 смартфон и явные купюры видны **в разных монтажных планах**.
DTiQ содержит еду, но одновременные банкноты неразличимы; сохранена ссылка, файл не включён.

**План, CSV и инструмент готовы; полный набор 32 нарушений + 8 чистых двойников
на четырёх заданных ракурсах пока не собран. Квалифицированных EXT — 0 из 40.**
Не подтверждены открытый сейф при уходе/клиенте, неправильная инкассация, скрытый
пересчёт и полный комплект остальных событий с парными clean. `coverage.csv` показывает
эти пробелы отдельно от найденных аналогов. Открытый кассовый ящик не считается сейфом;
сокрытие рук не доказывает пересчёт под столом. Исходные классы theft/Normal/Burglary
не преобразуются автоматически в V01–V08.

Файлы имеют префикс `CAND-*`, а не `EXT-*`. Монтаж исходника, две камеры рядом и
несколько отрезков одного события не считаются независимыми ракурсами или clean-парами.
`source_group` удерживает связанные записи в одном разбиении. Восемь DVIDS — восемь
разных публикаций, но это не гарантия независимости физических мест/сессий.

Условия источников: страницы восьми DVIDS явно указывают PUBLIC DOMAIN; оригинальные
авторы записаны в `credit`, действует [уведомление DVIDS](https://www.dvidshub.net/about/copyright).
Для UCF, INRIA, iLinkPro и перепубликации Instablog право использования датасетом
не установлено. Coverr исключён из-за прямого запрета AI/dataset use. Ранее собранные
8 MEVA с CC BY 4.0 остаются отдельными примерами общих действий.

The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW endorsement.

Исходная частота новых видео записана в `native_fps`: перевод в 30 fps повторяет или
отбрасывает кадры и не восстанавливает детали движения. Зеркалирование, ускорение,
зацикливание и генерация событий не применялись. Звук и метаданные контейнера удалены;
разрешение при необходимости уменьшено до ширины 1280, без увеличения. Это не съёмка iPhone.
В Git находятся только CSV и README; видео и оригиналы сохранены локально вне репозитория.

Ниже сохранён согласованный план, определения нарушений, шаблон и скрипт для 40 целевых
видео. Он задаёт требования к отбору и не означает, что съёмка уже выполнена.

Согласованный объём: **32 нарушения + 8 чистых двойников**. Все V01–V08 запланированы с четырёх
ракурсов; каждый сценарий имеет один чистый двойник с тем же ракурсом, людьми и реквизитом.
На каждом ракурсе 8 нарушений и 2 чистых ролика. Это компромисс лимита 40: полные пары
«нарушение + clean» для каждого из 8 × 4 сочетаний дали бы 64 ролика.

Основа плана — условия задачи команды и сообщённое участником разрешение FAQ на собственную съёмку.
План описывает постановочные действия; V01–V08 — локальные идентификаторы этой дополнительной выборки.
Они не объявляются нормативными кодами банка и не подменяют каталог нарушений сервиса.

## Файлы и быстрый порядок работы

В Git входят только README и CSV в `datasets/extra/`: `shooting_plan.csv`, `labels.csv`,
`external_candidates.csv`, `source_audit.csv`, `coverage.csv`.
Видео, журнал переименования и извлечённый скрипт хранятся **вне Git-репозитория**.
Ветка — `extra-data`, база PR — `main`.

1. До съёмки откройте таблицу ниже на телефоне. Один номер EXT — один удачный дубль.
2. Снимайте блоками по ракурсам. После каждого удачного дубля запишите исходное имя iPhone
   (`IMG_1234.MOV`) в `source_file` соответствующей строки `shooting_plan.csv`.
   Если имя недоступно на телефоне, ведите заметку «EXT → время записи / номер дубля» и
   сопоставьте с оригиналом после переноса. Не угадывайте по алфавитной сортировке файлов.
3. Передайте оригиналы без обрезки и перекодирования в отдельную папку вне репозитория.
   Брак храните отдельно, а в плане укажите ровно один выбранный файл на EXT.
4. Запустите проверку без `--apply`, затем переименование с `--apply` (команды ниже).
5. Просмотрите каждый ролик, заполните реальные `t_sec`, смените `status=planned` на
   `status=reviewed` и запустите проверку с `--labels`.

**Сейчас CSV — план и шаблон, а не размеченные наблюдения.** Все `t_sec` намеренно пустые.
Нельзя подавать строки `status=planned` в обучение/оценку как готовую разметку.

## Подготовка площадки и камеры

- iPhone неподвижно закреплён на шкафу/полке на высоте 2–2.5 м; горизонтальная запись
  `.mov`, настройка 30 fps, ориентир 1080p. Сделайте пробный 20-секундный дубль и проверьте
  экспорт через ffprobe. Если доступно автоматическое снижение fps, отключите его.
- Длительность каждого итогового файла 15–60 с. В плане целимся в 30 или 40 с: остаётся
  запас относительно границ. Оригиналы сохраняйте без монтажа; звук для разметки не нужен.
- В кадре: вся столешница, руки, рабочая зона за столом, дверца сейфа, входная дверь и
  место сумки. Дверь помещения и дверца сейфа — разные объекты. Для V04 нужен видимый
  порог настоящей двери или отдельной постановочной двери; не заменяйте его дверцей сейфа.
- Стол, ноутбук, телефон, игрушечные купюры, зелёная сумка, закрываемый металлический ящик
  или тумба, счётчик/коробка. Коробку-счётчик поставьте так, чтобы она не закрывала руки.
  В notes укажите `counter=box`, если используется замена.
- Люди: кассир, клиент, инкассатор. Клиент играет постороннего в V01/V04. Третий участник
  следит за записью, когда не играет инкассатора. Выберите постоянную рабочую зону кассира.
- Разметьте для актёров границу рабочей зоны и порог; не показывайте таблички «violation»
  или «clean» в ролике. Ноутбук показывает фиктивные данные; используются игрушечные деньги.

### Четыре разных ракурса

Лево/право определяются **со стороны сидящего кассира, который смотрит на клиента**.
Клиент находится перед столом. Высота каждого ракурса 2–2.5 м, взгляд сверху вниз.
Позиции не зеркалить программно: физически переставлять камеру.

| angle | Положение | Проверка перед блоком |
|---|---|---|
| left | Передний левый угол стола, диагональ около 45° к линии кассир–клиент | Видны руки, сейф и вход; 10 роликов |
| right | Передний правый угол, противоположная диагональ около 45° | Не просто отражение left; 10 роликов |
| side | Слева строго сбоку, около 90° к линии кассир–клиент | Профиль стола; видно, когда руки уходят ниже столешницы; 10 роликов |
| front | Почти со стороны клиента, около 0–15° к линии кассир–клиент | Ноутбук не закрывает деньги и руки; 10 роликов |

Ракурс организаторов «сзади-сбоку сверху» не входит в эти четыре. Перед каждым блоком
перепроверьте видимость дверцы и порога. Небольшую перестановку реквизита между блоками
запишите в notes. Внутри пары violation/clean сохраняйте положение камеры, свет, одежду,
реквизит и видимые предметы: различаться должно действие, а не фон.

## Режиссура и момент начала нарушения

Начало каждого дубля: 4–7 с спокойной сцены. Затем 3–6 с подхода/подготовки, 8–15 с
целевого действия и 5–10 с естественного завершения. Для 40-секундных дублей добавьте
спокойную работу до и после. Варьируйте начало действия примерно в диапазоне 6–12 с;
это подсказка для постановки, **не значение `t_sec`**. Не начинайте всегда в одну секунду.

Перед следующим дублем восстановите исходное состояние. Снимайте только одно целевое
нарушение: например, при V02 сейф уже закрыт; при V03 денег на столе нет; при V04
посторонний ещё не пересекает границу за столом. Clean проверяется на отсутствие всех
восьми сценариев, а не только своего парного Vxx.

| type_id | Что разыгрываем | Первый наблюдаемый момент для `t_sec` | Как выглядит чистый двойник |
|---|---|---|---|
| V01 | Посторонний входит за стол | Первый кадр пересечения посторонним заранее определённой границы рабочей зоны | Клиент подходит, но остаётся с клиентской стороны |
| V02 | Кассир уходит; купюры остаются | Первый кадр, когда кассир вышел из рабочей зоны, а деньги ещё видны на столе | Все деньги убраны в закрытый сейф до ухода |
| V03 | left/side: сейф открыт и кассир отошёл; right/front: открыт при клиенте | Для ухода — выход из рабочей зоны при открытой дверце; для клиента — открытие дверцы при уже присутствующем клиенте | Сейф закрыт до отхода; клиента во время открытия нет |
| V04 | Через открытую дверь входит посторонний | Первый кадр пересечения порога посторонним; одна открытая дверь без входа не является целью этого дубля | Посетитель ждёт снаружи за закрытой дверью; входа нет |
| V05 | Использование мобильного при деньгах | Первое наблюдаемое действие с телефоном: касание экрана после поднятия либо поднесение к уху, пока деньги видны | Телефон виден и лежит без использования; работа идёт на ноутбуке |
| V06 | left/side: еда; right/front: питьё у денег | Первый видимый укус или контакт чашки с губами при открытых деньгах | Деньги убраны в закрытый сейф до питья; чашка остаётся той же |
| V07 | Сумка оставлена без контроля / уход до приёма | Начало отхода инкассатора после оставления сумки без приёма кассиром | Сумка остаётся под контролем инкассатора до явного приёма кассиром |
| V08 | Пересчёт уходит под стол | Первый кадр, где купюры и рабочие движения рук уходят из видимой зоны пересчёта | Те же купюры и движения над столом, обе руки видны |

V07 использует наблюдаемую постановочную процедуру «показать сумку → кассир принимает →
инкассатор уходит». Это не утверждение о полной банковской процедуре. Для V08 покажите
купюры до ухода под стол и возвращение после; не размечайте скрытый пересчёт только по догадке.
Если действие заслонено человеком/ноутбуком и начало нельзя уверенно определить, переснимите дубль.

## Таблица съёмки: 40 файлов

Целевые секунды ниже — только план длительности. Номер EXT остаётся постоянным при пересъёмке.
Чистый двойник стоит сразу после соответствующего нарушения; остальные нарушения дополняют
покрытие ракурсов. Пары имеют одинаковый `pair_id` в CSV.

| Файл | Тип | angle | label | План, с | Действие |
|---|---|---|---|---:|---|
| EXT-001.mov | V01 | left | violation | 30 | Клиент из клиентской зоны пересекает границу и заходит за стол кассира; дверь закрыта |
| EXT-002.mov | V01 | left | clean | 30 | Клиент подходит к границе рабочей зоны; остаётся с клиентской стороны; кассир обслуживает его |
| EXT-003.mov | V02 | left | violation | 30 | Кассир оставляет купюры на столе и выходит из рабочей зоны; сейф закрыт; клиента нет |
| EXT-004.mov | V03 | left | violation | 40 | Кассир оставляет сейф открытым и отходит; клиента нет; денег на столе нет |
| EXT-005.mov | V04 | left | violation | 30 | Посторонний проходит через открытую дверь в кассовую зону; за стол не заходит |
| EXT-006.mov | V05 | left | violation | 30 | При видимых купюрах кассир берёт телефон; листает экран или подносит телефон к уху |
| EXT-007.mov | V05 | left | clean | 30 | Телефон лежит на столе без использования; кассир считает видимые деньги и работает на ноутбуке |
| EXT-008.mov | V06 | left | violation | 30 | Кассир ест рядом с открытыми купюрами; остаётся за столом |
| EXT-009.mov | V07 | left | violation | 40 | Инкассатор ставит зелёную сумку и отходит без передачи; кассир её не принимает; сумка без контроля |
| EXT-010.mov | V08 | left | violation | 30 | Кассир показывает купюры на столе; опускает их с руками ниже столешницы и продолжает пересчёт |
| EXT-011.mov | V01 | right | violation | 30 | Клиент из клиентской зоны пересекает границу и заходит за стол кассира; дверь закрыта |
| EXT-012.mov | V02 | right | violation | 30 | Кассир оставляет купюры на столе и выходит из рабочей зоны; сейф закрыт; клиента нет |
| EXT-013.mov | V02 | right | clean | 30 | Кассир убирает все купюры в сейф; закрывает его; затем отходит; стол пуст |
| EXT-014.mov | V03 | right | violation | 40 | Кассир открывает сейф при уже присутствующем клиенте; остаётся на месте; денег на столе нет |
| EXT-015.mov | V04 | right | violation | 30 | Посторонний проходит через открытую дверь в кассовую зону; за стол не заходит |
| EXT-016.mov | V05 | right | violation | 30 | При видимых купюрах кассир берёт телефон; листает экран или подносит телефон к уху |
| EXT-017.mov | V06 | right | violation | 30 | Кассир пьёт рядом с открытыми купюрами; остаётся за столом |
| EXT-018.mov | V06 | right | clean | 30 | Кассир убирает все деньги в закрытый сейф; затем пьёт за тем же столом; возвращается к работе |
| EXT-019.mov | V07 | right | violation | 40 | Инкассатор оставляет зелёную сумку и уходит до приёма кассиром; видимого подтверждения передачи нет |
| EXT-020.mov | V08 | right | violation | 30 | Кассир показывает купюры на столе; опускает их с руками ниже столешницы и продолжает пересчёт |
| EXT-021.mov | V01 | side | violation | 30 | Клиент из клиентской зоны пересекает границу и заходит за стол кассира; дверь закрыта |
| EXT-022.mov | V02 | side | violation | 30 | Кассир оставляет купюры на столе и выходит из рабочей зоны; сейф закрыт; клиента нет |
| EXT-023.mov | V03 | side | violation | 40 | Кассир оставляет сейф открытым и отходит; клиента нет; денег на столе нет |
| EXT-024.mov | V03 | side | clean | 40 | Кассир открывает сейф без клиента; закрывает дверцу до отхода; клиент не появляется |
| EXT-025.mov | V04 | side | violation | 30 | Посторонний проходит через открытую дверь в кассовую зону; за стол не заходит |
| EXT-026.mov | V05 | side | violation | 30 | При видимых купюрах кассир берёт телефон; листает экран или подносит телефон к уху |
| EXT-027.mov | V06 | side | violation | 30 | Кассир ест рядом с открытыми купюрами; остаётся за столом |
| EXT-028.mov | V07 | side | violation | 40 | Инкассатор ставит зелёную сумку и отходит без передачи; кассир её не принимает; сумка без контроля |
| EXT-029.mov | V07 | side | clean | 40 | Инкассатор держит зелёную сумку до явного приёма кассиром; кассир сразу берёт её под контроль |
| EXT-030.mov | V08 | side | violation | 30 | Кассир показывает купюры на столе; опускает их с руками ниже столешницы и продолжает пересчёт |
| EXT-031.mov | V01 | front | violation | 30 | Клиент из клиентской зоны пересекает границу и заходит за стол кассира; дверь закрыта |
| EXT-032.mov | V02 | front | violation | 30 | Кассир оставляет купюры на столе и выходит из рабочей зоны; сейф закрыт; клиента нет |
| EXT-033.mov | V03 | front | violation | 40 | Кассир открывает сейф при уже присутствующем клиенте; остаётся на месте; денег на столе нет |
| EXT-034.mov | V04 | front | violation | 30 | Посторонний проходит через открытую дверь в кассовую зону; за стол не заходит |
| EXT-035.mov | V04 | front | clean | 30 | Дверь закрыта; посетитель подходит и ждёт снаружи; в кассовую зону не входит |
| EXT-036.mov | V05 | front | violation | 30 | При видимых купюрах кассир берёт телефон; листает экран или подносит телефон к уху |
| EXT-037.mov | V06 | front | violation | 30 | Кассир пьёт рядом с открытыми купюрами; остаётся за столом |
| EXT-038.mov | V07 | front | violation | 40 | Инкассатор оставляет зелёную сумку и уходит до приёма кассиром; видимого подтверждения передачи нет |
| EXT-039.mov | V08 | front | violation | 30 | Кассир показывает купюры на столе; опускает их с руками ниже столешницы и продолжает пересчёт |
| EXT-040.mov | V08 | front | clean | 30 | Кассир пересчитывает купюры на открытой части стола; купюры и обе руки видны |

Чистые пары: V01 и V05 — left; V02 и V06 — right; V03 и V07 — side; V04 и V08 — front.
План содержит 21 мин 40 с итогового материала. С настройкой, сменой ракурсов, сбросом сцен
и пересъёмками заложите примерно 2–3 часа — это оценка организации съёмки, не норматив.

## Разметка `labels.csv`

Точный заголовок: `file,label,t_sec,type_id,angle,notes`. UTF-8, разделитель запятая,
десятичный разделитель точка. Одна строка на один файл. Поля с запятыми заключаются в кавычки.

| Поле | Правило |
|---|---|
| file | Итоговый basename `EXT-001.mov` … `EXT-040.mov`; не абсолютный путь |
| label | Только `violation` или `clean`; сейчас указано запланированное значение |
| t_sec | Начало нарушения от первого отображаемого кадра оригинального файла в секундах; например `8.433`. Для clean всегда пусто, не `0` и не `-1` |
| type_id | `V01`…`V08`. Для clean сохраняется тип парного сценария, но это НЕ наличие нарушения; целевую метку определяет label |
| angle | Только `left`, `right`, `side`, `front` |
| notes | Метаданные через `;`: статус, пара, реквизит, наблюдаемое действие, ограничения видимости |

Просмотрите видео полностью, затем покадрово установите начало по таблице выше. Время —
относительно начала воспроизведения, не часы iPhone, не момент начала записи второго дубля,
не пик действия. При строго постоянных 30 fps кадр с нулевым индексом n даёт `n/30`;
при 29.97 fps используйте реальные временные метки/таймлайн, а не деление на 30.
Переименование не меняет таймкоды, перекодирования нет.

Пусть второй участник независимо проверит label и начало. При расхождении более 0.5 с
пересмотрите кадры вместе; 0.5 с здесь — рабочий порог контроля разметки, не допуск жюри.
После просмотра замените `status=planned` на `status=reviewed`, допишите, например,
`onset=кассир пересёк границу;reviewer=участник2;counter=box`.
Для сомнительных дублей ставьте `status=needs_review` и переснимайте: скрипт их не примет
как готовую разметку. Пустое t_sec у violation означает незавершённую разметку.

Если случилось второе нарушение, переснимите: этот CSV описывает один целевой тип на файл.
Не присваивайте clean автоматически по сценарию. Если удачный дубль фактически отличается
от плана, согласованно исправьте обе CSV; баланс 32/8 и пары должны сохраниться.
Локальные `clean` и `t_sec` не равны полям API сервиса: при интеграции явно сопоставьте
`clean` с принятым в сервисе отрицательным классом и Vxx с его каталогом.

## Проверка и переименование после съёмки

Нужны Python 3.11+ и `ffprobe` из FFmpeg. Источники: [загрузка FFmpeg](https://ffmpeg.org/download.html)
и [документация ffprobe](https://ffmpeg.org/ffprobe.html). `ffprobe` читает метаданные и
временные метки в JSON; скрипт не использует облачные сервисы и не перекодирует видео.

Храните видео вне любого Git checkout, например в отдельной папке `forte-extra-media`
рядом с репозиторием. Сделайте резервную копию оригиналов до переименования. В текущем
`main` нет корневого `.gitignore`; менять чужие корневые файлы для этой задачи не требуется.
Защита от случайного коммита обеспечивается расположением видео вне Git и точным `git add`.

Скрипт хранится ниже, чтобы в PR оставались только CSV и README. Извлечение **из корня
репозитория**, работает в PowerShell и обычном shell с Python:

```shell
python -c "from pathlib import Path; s=Path('datasets/extra/README.md').read_text(encoding='utf-8').splitlines(); p=Path('../extra-tools'); p.mkdir(exist_ok=True); (p/'prepare_extra.py').write_text('\n'.join(s[s.index('<!-- EXTRA_SCRIPT_BEGIN -->')+2:s.index('<!-- EXTRA_SCRIPT_END -->')-1])+'\n', encoding='utf-8')"
```

Заполните все 40 значений `source_file` реальными именами оригиналов (только имя файла,
без подпапок). Пути с пробелами заключайте в кавычки.

```shell
python ../extra-tools/prepare_extra.py ../forte-extra-media --plan datasets/extra/shooting_plan.csv
python ../extra-tools/prepare_extra.py ../forte-extra-media --plan datasets/extra/shooting_plan.csv --apply
```

Если `ffprobe` не находится через PATH, к команде добавьте `--ffprobe "C:/путь/ffprobe.exe"`.
Первый запуск ничего не записывает. Второй переименовывает только после успешной проверки
**всех** 40 файлов и отсутствия конфликтов. Имя EXT берётся из плана, не из порядка каталога.

Проверки: 15 ≤ длительность ≤ 60 с; средний и номинальный fps в пределах 30 ± 0.1
(принимается 30000/1001 ≈ 29.97); последовательные PTS каждого кадра, отклонение шага от
`1/avg_fps` не более 2 мс; согласие длительности декодированных кадров с метаданными
в пределах 0.1 с. VFR, пропуски кадров, ошибки декодирования, отсутствующие метки времени,
неизвестный fps и файлы без видео требуют проверки/пересъёмки и блокируют весь запуск.
Эти пороги — локальные критерии качества датасета. `.MOV`/`.mov` принимаются; результат — `.mov`.
Проверка контейнера принимает семейство MOV/QuickTime; исходный формат задаётся съёмкой на iPhone.

`--apply` создаёт вне Git `rename_log.csv`: исходное и новое имя, fps, длительность,
число кадров, SHA-256 и события `planned`/`renamed`/`already_named`. Журнал записывается
до первого переименования и после каждого файла. Существующие видео и журнал не затираются.
Не редактируйте папку видео параллельно запуску. На Windows используется rename без замены;
на POSIX файловая система должна поддерживать hard links (NTFS/APFS/ext4 обычно подходят).

Если запуск оборвался при переименовании, изменения могут быть частичными: не запускайте
`--apply` вслепую повторно. Сопоставьте имена и SHA-256 с журналом. Для успешно переименованных
строк обновите `source_file` на соответствующее EXT-имя, для остальных сохраните оригинал.
Сохраните журнал под отдельным именем, затем повторите обычную проверку и `--apply`.
Если остались обе копии файла, сначала сверяйте хэши и выбирайте нужный источник вручную.

После штатного переименования обновите `source_file` на итоговые имена, сохранив журнал
как связь с оригиналами. Заполните разметку и проверьте её:

```shell
python ../extra-tools/prepare_extra.py ../forte-extra-media --plan datasets/extra/shooting_plan.csv --labels datasets/extra/labels.csv
```

Этот режим проверит реальные файлы и потребует `status=reviewed` во всех notes,
непустое конечное `t_sec` в пределах файла у violation и пустое значение у clean.
План, углы, типы, метки и имена в двух CSV должны совпадать. Автоматически определить
нарушение или заполнить фактический таймкод этот технический скрипт не пытается.

<!-- EXTRA_SCRIPT_BEGIN -->
```python
"""Validate an explicit shooting manifest, then rename MOV files without overwriting."""
import argparse
import csv
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from fractions import Fraction
import re
import subprocess
import sys

ANGLES = {"left", "right", "side", "front"}
LABEL_FIELDS = ["file", "label", "t_sec", "type_id", "angle", "notes"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        require(rows and all(None not in row and None not in row.values() for row in rows),
                f"Malformed or empty CSV: {path}")
        return reader.fieldnames, rows


def number(value):
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc
    require(math.isfinite(result), f"Non-finite number: {value}")
    return result


def probe(path, executable):
    cmd = [executable, "-v", "error", "-select_streams", "v:0", "-show_frames",
           "-show_entries", "stream=avg_frame_rate,r_frame_rate,duration:"
           "format=duration,format_name:frame=best_effort_timestamp_time", "-of", "json", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=180, check=False)
    require(result.returncode == 0 and not result.stderr.strip(),
            f"ffprobe failed for {path.name}: {result.stderr.strip()[:400]}")
    data = json.loads(result.stdout)
    require(len(data.get("streams", [])) == 1, f"No video stream: {path.name}")
    stream = data["streams"][0]
    require("mov" in data.get("format", {}).get("format_name", "").split(","),
            f"Not a MOV/QuickTime-family container: {path.name}")
    duration_raw = stream.get("duration")
    if duration_raw in (None, "N/A"):
        duration_raw = data.get("format", {}).get("duration")
    duration = number(duration_raw)
    avg = number(stream["avg_frame_rate"])
    nominal = number(stream["r_frame_rate"])
    require(15 <= duration <= 60, f"Duration {duration:.3f}s outside [15, 60]: {path.name}")
    require(abs(avg - 30) <= 0.1 and abs(nominal - 30) <= 0.1,
            f"FPS avg={avg:.4f}, nominal={nominal:.4f}; expected 30 +/- 0.1: {path.name}")
    frames = data.get("frames", [])
    require(len(frames) >= 2 and all("best_effort_timestamp_time" in f for f in frames),
            f"Missing frame timestamps: {path.name}")
    pts = [number(f["best_effort_timestamp_time"]) for f in frames]
    steps = [b - a for a, b in zip(pts, pts[1:])]
    require(all(step > 0 for step in steps), f"Non-monotonic frame timestamps: {path.name}")
    max_jitter = max(abs(step - 1 / avg) for step in steps)
    require(max_jitter <= 0.002, f"VFR / dropped frames (jitter {max_jitter:.6f}s): {path.name}")
    pts_duration = pts[-1] - pts[0] + 1 / avg
    require(abs(pts_duration - duration) <= 0.1,
            f"Decoded frames disagree with duration: {path.name}")
    return {"duration_sec": round(duration, 6), "avg_fps": round(avg, 6),
            "nominal_fps": round(nominal, 6), "frames": len(frames),
            "max_jitter_sec": round(max_jitter, 6)}


def load_plan(plan_path, media):
    fields, rows = read_csv(plan_path)
    required = {"file", "source_file", "label", "type_id", "angle"}
    require(required <= set(fields), f"Plan needs columns: {sorted(required)}")
    require(len(rows) == 40, "Expected exactly 40 rows in the shooting plan")
    require({r["file"] for r in rows} == {f"EXT-{i:03d}.mov" for i in range(1, 41)},
            "Plan must contain unique EXT-001.mov ... EXT-040.mov")
    require(Counter(r["label"] for r in rows) == {"violation": 32, "clean": 8},
            "Plan must contain 32 violations and 8 clean twins")
    require({(r["type_id"], r["angle"]) for r in rows if r["label"] == "violation"}
            == {(f"V{i:02d}", angle) for i in range(1, 9) for angle in ANGLES},
            "Violations must cover every type and angle exactly once")
    clean = [r for r in rows if r["label"] == "clean"]
    require(Counter(r["type_id"] for r in clean) == {f"V{i:02d}": 1 for i in range(1, 9)}
            and Counter(r["angle"] for r in clean) == {angle: 2 for angle in ANGLES},
            "Clean twins need one per scenario and two per angle")
    sources = set()
    jobs = []
    for row in rows:
        name = row["source_file"]
        require(name and not any(c in name for c in '/\\:') and Path(name).suffix.lower() == ".mov",
                f"Set a flat original MOV filename for {row['file']}: {name!r}")
        require(name.casefold() not in sources, f"Duplicate source_file: {name}")
        sources.add(name.casefold())
        require(row["label"] in {"violation", "clean"} and row["angle"] in ANGLES
                and re.fullmatch(r"V0[1-8]", row["type_id"]), f"Invalid plan metadata: {row['file']}")
        require(not name.upper().startswith("EXT-") or name == row["file"],
                f"Reserved EXT name as source (rename chains are forbidden): {name}")
        source = media / name
        target = media / row["file"]
        require(source.is_file() and not source.is_symlink() and source.resolve().parent == media,
                f"Missing or unsafe source: {source}")
        require(source == target or not target.exists(), f"Target already exists: {target}")
        stat = source.stat()
        jobs.append((row, source, target, (stat.st_size, stat.st_mtime_ns)))
    return jobs


def check_labels(path, jobs, metrics):
    fields, rows = read_csv(path)
    require(fields == LABEL_FIELDS, f"Expected labels header: {','.join(LABEL_FIELDS)}")
    by_file = {row["file"]: row for row in rows}
    require(len(rows) == len(by_file) == len(jobs) and set(by_file) == set(metrics),
            "Labels must match all 40 plan files exactly once")
    for plan, _, _, _ in jobs:
        row = by_file[plan["file"]]
        require(all(row[key] == plan[key] for key in ("label", "type_id", "angle")),
                f"Plan/labels mismatch: {row['file']}")
        require("status=reviewed" in row["notes"].split(";"), f"Not reviewed: {row['file']}")
        if row["label"] == "clean":
            require(row["t_sec"] == "", f"Clean t_sec must be empty: {row['file']}")
        else:
            require(row["t_sec"] != "", f"Missing onset: {row['file']}")
            require(0 <= number(row["t_sec"]) < metrics[row["file"]]["duration_sec"],
                    f"Onset outside video: {row['file']}")


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rename_no_replace(source, target):
    # Windows rename fails if destination exists. On POSIX, reserve via a hard link.
    if os.name == "nt":
        os.rename(source, target)
    else:
        os.link(source, target)
        source.unlink()


def apply_renames(jobs, metrics, journal):
    # Refuse changed inputs or new target collisions after the potentially long probes.
    for _, source, target, initial in jobs:
        stat = source.stat()
        require((stat.st_size, stat.st_mtime_ns) == initial, f"Source changed during QA: {source}")
        require(source == target or not target.exists(), f"Target appeared during QA: {target}")
    require(not journal.exists(), f"Journal already exists; inspect it before retrying: {journal}")
    records = []
    for row, source, _, _ in jobs:
        records.append({"source_file": source.name, "file": row["file"],
                        **metrics[row["file"]], "sha256": fingerprint(source), "status": "planned"})
    with journal.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
        for (_, source, target, _), record in zip(jobs, records):
            if source == target:
                state = "already_named"
            else:
                rename_no_replace(source, target)
                state = "renamed"
            writer.writerow({**record, "status": state})
            handle.flush()
            os.fsync(handle.fileno())
    print(f"APPLIED: {len(jobs)} files; journal: {journal}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Also require completed, reviewed annotation")
    parser.add_argument("--ffprobe", default="ffprobe", help="Executable name or full path")
    parser.add_argument("--apply", action="store_true", help="Rename only after every check passes")
    args = parser.parse_args(argv)
    media = args.media_dir.resolve(strict=True)
    require(media.is_dir(), "media_dir must be a directory")
    require(not any((p / ".git").exists() for p in (media, *media.parents)),
            "Keep videos in a directory outside every Git checkout")
    jobs = load_plan(args.plan, media)
    metrics, errors = {}, []
    for row, source, target, _ in jobs:
        try:
            info = probe(source, args.ffprobe)
            metrics[row["file"]] = info
            print(f"OK {source.name} -> {target.name} | {info['duration_sec']:.3f}s | "
                  f"{info['avg_fps']:.4f} fps | {info['frames']} frames")
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{source.name}: {exc}")
    require(not errors, "QA failed; nothing renamed:\n" + "\n".join(errors))
    if args.labels:
        check_labels(args.labels, jobs, metrics)
    if args.apply:
        apply_renames(jobs, metrics, media / "rename_log.csv")
    else:
        print("DRY RUN passed; nothing written. Use --apply to rename.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
```
<!-- EXTRA_SCRIPT_END -->

## Передача команде и оценка

Передайте команде CSV и путь к отдельному хранилищу оригиналов/итоговых MOV; сами видео
не помещайте в репозиторий и PR. Для обучения/валидации держите оба члена пары в одном
разбиении. Одна запись, её пересъёмки, нарезки и близкие дубли одной сцены не должны
попадать одновременно в train и test. Для проверки переноса ракурса можно исключать целый
angle из обучения; по возможности проверять на отдельно разыгранной сессии. Не объявляйте
оценку на знакомой сцене независимой оценкой переноса на новое место.

В каждом оставленном ракурсе только 2 clean: число ложных срабатываний на них показывайте
как FP / количество clean с указанием малого объёма. Этот набор не гарантирует отсутствие
ложных срабатываний и не заменяет финальный тест организаторов.

После каждого законченного изменения CSV/README выполняйте из корня репозитория:

```shell
git add -- datasets/extra/README.md datasets/extra/shooting_plan.csv datasets/extra/labels.csv datasets/extra/external_candidates.csv datasets/extra/source_audit.csv datasets/extra/coverage.csv
git diff --cached --stat
git commit -m "data(extra): update shooting plan or reviewed labels"
git push origin extra-data
```

Проверьте список подготовленных файлов перед commit: должны быть только эти шесть путей.
Не используйте `git add .` для импорта снятого материала. PR направляется в `main`;
слияние в рамках этой задачи не выполняется.
