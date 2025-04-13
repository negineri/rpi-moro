# moro

工具箱

## 概要

このプロジェクトは、通常は一度きりで使用され、独自のプロジェクトを作成するほど重要ではないスクリプトを集めたものです。これらのスクリプトをサブコマンド形式で集約し、将来的に再利用できるようにすることを目的としています。

## インストール

```text
pip install git+ssh://git@github.com/negineri/moro.git
```

## 開発

### 必要条件

- uv

### セットアップ

```text
uv run moro
pre-commit install
```

### 開発心得

- Visual Studio Code での開発を想定しています
- `tox p`を定期的に実行すると良いです
  - 一般に pytest は処理が重いため、pre-commit には含めていません。
