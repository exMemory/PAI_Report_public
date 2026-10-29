# Unitree G1: サッカー PK の階層制御

Physical AI 応用1の提出用コードです。MuJoCo 上の Unitree G1 に対し、
学習済み RL 歩行ポリシーと共通キック動作を下位制御として使用し、
上位の参照ルール方策 / Gemini / Jev を比較します。
最終ルールは **60秒以内・キック回数無制限・転倒したら終了** です。

## 提出範囲

- [src/](src/): 最終版の閉ループ制御、実 API アダプター、録画処理の7ファイル。
- [diag/](diag/): 最終動画で使用した試行単位の再試行と5画面グリッド合成のみ。
- [assets/unitree_g1/](assets/unitree_g1/): 使用する MJCF、参照先の27メッシュ、
  学習済みポリシー、ライセンス。追加のモデルダウンロードは不要です。
- [tests/](tests/): 最終版のオフライン回帰テスト。
- VLA モック、旧 PD 制御のデモ、開発メモ、課題ガイドライン、レポート、
  既存動画、ログ、仮想環境、認証情報は含みません。
  PDF は課題フォーム、動画は YouTube に別途提出してください。

参照方策はローカル実行のみです。Gemini / Jev を選ぶと実 API を呼び出し、
料金が発生する場合があります。AI に渡すのは数値状態の JSON であり、
画像入力や VLA 推論ではありません。テストの API クライアントはモックです。
RL ポリシーの推論は NumPy で実装されているため PyTorch は不要です。

## セットアップ

Windows / PowerShell、Python 3.11 を推奨します。以下は必ずこのフォルダーを
カレントディレクトリにして実行してください（ポリシーのパスが相対指定です）。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

録画には OpenGL が使える環境と日本語フォントが必要です。
Windows は Meiryo または Yu Gothic、Linux は Noto Sans CJK
(`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`) を使用します。
FFmpeg は `imageio[ffmpeg]` の依存パッケージで導入します。

## API なしの物理実行

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from ai_pk_runner import RuleBasedDecider, run_ai_episode; from dataclasses import asdict; import json; print(json.dumps(asdict(run_ai_episode(RuleBasedDecider(), max_time=60.0, stop_on_fall=True)), indent=2))"
```

MuJoCo と同梱 RL ポリシーを実際に動かし、得点・転倒・キック回数などを出力します。
`ai_pk_runner.py` 単独の旧 CLI は既定の終了条件が異なるため、
最終ルールの検証には上記コマンドを使用してください。

## 5試行の録画

```powershell
.\.venv\Scripts\python.exe src\render_ai_pk_video.py --provider reference --trials 5 --vary-ball
```

横位置を -0.30 / -0.15 / 0 / +0.15 / +0.30 m に変えて各試行を録画します。
毎回新しい `outputs\ai_pk_closed_loop_videos\<日時>\five_trials` が作られ、
各試行の MP4 / JSON、集計 JSON、結果画像、連結動画が保存されます。
連結動画は最大約5分になるため、提出用の1分グリッドは次のように作ります。

```powershell
.\.venv\Scripts\python.exe diag\build_grid_video.py --batch-dir "outputs\ai_pk_closed_loop_videos\<日時>\five_trials" --title "Reference" --output "outputs\reference_grid.mp4"
```

`<日時>` は実際に生成されたフォルダー名に置き換えてください。
グリッドは 1920x960 / 30 fps / 60秒で、早く終了した試行は最終フレームを保持します。
出力先には未使用のファイル名を指定してください。既存ファイルは拒否します。
各方式のグリッド3本を動画編集ソフトなどで連結すると約3分になります。

### Gemini / Jev の実 API

プロジェクト直下にローカル専用の `.env` を作り、利用するキーだけを設定します。
プロセス環境変数での設定も可能です。

```text
GEMINI_API_KEY=（自分のキーをローカルで設定）
OPENROUTER_API_KEY=（自分のキーをローカルで設定）
```

Gemini は `GOOGLE_API_KEY` も利用できます。キーをコードや README に書かないでください。
`.env` は公開対象外です。既定モデルの利用権限・提供状況は各サービスに依存します。

```powershell
.\.venv\Scripts\python.exe src\render_ai_pk_video.py --provider gemini --trials 5 --vary-ball
.\.venv\Scripts\python.exe src\render_ai_pk_video.py --provider jev --trials 5 --vary-ball
```

既定は `gemini-robotics-er-2-preview` / `typesafe/jev-router` です。
`--model` で変更できますが、アダプターの出力契約に対応したモデルが必要です。
通信・出力検証のエラーは成功扱いにせず中断します。
試行単位の再試行が必要な場合だけ、以下を利用してください。

```powershell
.\.venv\Scripts\python.exe diag\record_trials_resilient.py --provider gemini --run-dir "outputs\gemini_retry_001" --vary-ball --max-attempts 3
```

初回は未使用のディレクトリ名を指定してください。
同じ引数で再実行すると、保存済み JSON がある試行をスキップして再開します。
別モデル・別条件には必ず別ディレクトリを使用してください。
再試行も API 呼び出しを消費し、失敗履歴は集計に記録されます。

## オフライン検証

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

API 契約・待ち時間・蹴り直しのテストはモックです。
摩擦テストは実際の MuJoCo の接触計算を使用し、動画テストはローカルで描画・
エンコード・全フレームのデコード検証を行います。実 API の疎通テストではありません。

整理時の検証には既存の Python 3.11.9 環境を使用しています。
主要バージョンは MuJoCo 3.16.0 / NumPy 2.4.6 / OpenCV 5.0.0.93 /
imageio 2.38.1 / imageio-ffmpeg 0.6.0 / Pillow 13.0.0 /
google-genai 2.31.0 / openai 3.35.1 です。
新規環境へのインストールと実 API 呼び出しは整理時には行っていません。

確認結果（2026-10-29）:

- オフラインテスト28件成功（動画テストで imageio の `ResourceWarning` が発生）。
- 参照方策を最終ルールで実行し、3回のキック、25.952秒で得点、転倒なし。
- 録画・再試行・グリッド合成の CLI と SDK の import、ローカル FFmpeg を確認。
- 公開対象49ファイル、約24.33 MiB。MJCF が参照する27メッシュの過不足なし。
- `.env`、出力、旧コードなど16パターンの除外と `git add --dry-run` を確認。
- Gemini / Jev の実 API と、新しい5試行の実録画・グリッド動画の全生成は未実施。

## 公開前の確認

このフォルダーは元の `.git` をコピーせず `main` ブランチで新規初期化しています。
**元リポジトリの履歴は削除・浄化していません。**
過去のファイルを公開したくない場合、元リポジトリを public にせず、
GitHub に別の空リポジトリを作成して、このフォルダーから PUSH してください。
すでに公開された履歴や第三者のコピーを、この整理で取り消すことはできません。

[.gitignore](.gitignore) は許可リスト方式です。今後追加するファイルは、
必要なものだけ例外に追加してください。`git add -f` は使用しないでください。
許可済みソース内にキーを書いた場合までは防げないため、差分も確認してください。

```powershell
git status --short --untracked-files=all
git add --dry-run .
# 表示された公開対象を確認してから実行
git add .
git diff --cached --stat
git commit -m "Initial submission" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
# URL は新規作成した空の提出用リポジトリに置き換える
git remote add origin https://github.com/<account>/<new-submission-repository>.git
git push -u origin main
```

公開時の Git 著者名・メールアドレスも確認し、必要なら GitHub の noreply
アドレスを設定してください。この整理ではコミット・リモート設定・PUSH は未実施です。

## ライセンス

同梱資産の出典とライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) と
[Unitree LICENSE](assets/unitree_g1/LICENSE) を参照してください。
既存のライセンス表示を保持しています。整理時のオンライン再調査は行っていません。
