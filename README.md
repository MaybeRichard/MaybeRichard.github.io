# MaybeRichard.github.io

用 Markdown 写作的极简博客，样式参考 [Atharva Naik 的博客](https://atharva-naik.github.io/blog/)。推送到 GitHub 后会自动构建并发布。

## 第一次部署（只需做一次）

1. 把这个文件夹里的所有文件（包括隐藏的 `.github` 文件夹）放进仓库根目录，推送到 `main` 分支。
2. 打开仓库的 **Settings → Pages**，把 **Source** 改成 **GitHub Actions**。
3. 到 **Actions** 标签页看构建进度。一两分钟后访问 https://mayberichard.github.io 就能看到网站。

如果你的默认分支不叫 `main`，把 `.github/workflows/deploy.yml` 里的 `main` 改成你的分支名。

## 日常写作

1. 在 `posts/` 里新建一个文件，比如 `posts/2026-11-20-my-post.md`：

   ```markdown
   ---
   title: 文章标题
   date: 2026-11-20
   tags: [研究]
   ---

   正文用 Markdown 写……
   ```

2. 推送到 GitHub，网站会自动更新。也可以直接在 GitHub 网页上新建文件、写完点 Commit。

文章地址是 `/blog/my-post/`（文件名去掉开头的日期）。

**有图片的文章**：建一个文件夹 `posts/2026-11-20-my-post/`，正文写在里面的 `index.md`，图片放在同一个文件夹，正文里用 `![说明](图.png "图注")` 引用。

**全部写法**（公式、引用、脚注、折叠块、系列文章等）请看示例文章 `posts/2026-11-16-writing-guide/index.md`，对照网页效果看最直观。

## 本地预览（可选）

```bash
pip install -r requirements.txt
python build.py --serve
```

打开 http://localhost:8000 ，改文件后刷新页面即可看到变化。预览时草稿（`draft: true`）也会显示，正式发布时不会。

## 文件说明

| 文件 | 作用 |
|:--|:--|
| `site.yml` | 网站设置：名字、语言、首页链接、系列、站外文章 |
| `about.md` | 首页的自我介绍 |
| `posts/` | 所有文章 |
| `static/` | 会原样复制到网站根目录的文件（如 `favicon.ico`、简历 PDF） |
| `theme/` | 样式（CSS），想调颜色或字体就改这里 |
| `build.py` | 构建脚本，一般不用动 |

两篇示例文章和 `site.yml` 里的示例系列 `blog-notes` 可以随时删掉。
