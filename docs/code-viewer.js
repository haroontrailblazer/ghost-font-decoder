(() => {
  const triggers = document.querySelectorAll(
    "[data-code-viewer], [data-repo-viewer], [data-text-viewer]"
  );
  if (!triggers.length) return;

  const cache = new Map();
  let activeTrigger;
  let activeText = "";
  let tripleQuote = null;
  let requestSequence = 0;
  let repositoryState = null;

  const dialog = document.createElement("dialog");
  dialog.className = "code-viewer";
  dialog.setAttribute("aria-labelledby", "content-viewer-title");
  dialog.innerHTML = `
    <div class="code-viewer-shell">
      <header class="code-viewer-header">
        <div class="code-viewer-title">
          <strong id="content-viewer-title">Source viewer</strong>
          <span class="code-viewer-meta"></span>
        </div>
        <button class="code-viewer-action code-viewer-copy" type="button">Copy</button>
        <a class="code-viewer-action code-viewer-open-source" href="#" target="_blank" rel="noopener">GitHub</a>
        <button class="code-viewer-action code-viewer-close" type="button" aria-label="Close source viewer">&times;</button>
      </header>
      <div class="code-viewer-body">
        <div class="code-viewer-status" role="status">Loading&hellip;</div>
      </div>
    </div>
  `;
  document.body.append(dialog);

  const title = dialog.querySelector("#content-viewer-title");
  const body = dialog.querySelector(".code-viewer-body");
  const meta = dialog.querySelector(".code-viewer-meta");
  const copyButton = dialog.querySelector(".code-viewer-copy");
  const githubLink = dialog.querySelector(".code-viewer-open-source");
  const closeButton = dialog.querySelector(".code-viewer-close");

  const escapeHtml = (value) => value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  const escapeAttribute = (value) => escapeHtml(value)
    .replaceAll('"', "&quot;");

  const keywords = new Set([
    "and", "as", "assert", "async", "await", "break", "case", "class",
    "continue", "def", "del", "elif", "else", "except", "False", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "match",
    "None", "nonlocal", "not", "or", "pass", "raise", "return", "True",
    "try", "while", "with", "yield"
  ]);
  const builtins = new Set([
    "abs", "all", "any", "bool", "bytes", "dict", "enumerate", "float",
    "int", "len", "list", "map", "max", "min", "open", "print", "range",
    "set", "str", "sum", "super", "tuple", "type", "zip"
  ]);

  function highlightPythonLine(line) {
    let html = "";
    let index = 0;
    let expectDefinition = false;

    const add = (value, className) => {
      const escaped = escapeHtml(value);
      html += className ? `<span class="${className}">${escaped}</span>` : escaped;
    };

    while (index < line.length) {
      if (tripleQuote) {
        const end = line.indexOf(tripleQuote, index);
        if (end === -1) {
          add(line.slice(index), "syntax-string");
          return html;
        }
        add(line.slice(index, end + 3), "syntax-string");
        index = end + 3;
        tripleQuote = null;
        continue;
      }

      const rest = line.slice(index);
      const tripleMatch = rest.match(/^(?:[rRuUbBfF]{0,2})("""|''')/);
      if (tripleMatch) {
        const delimiter = tripleMatch[1];
        const end = line.indexOf(delimiter, index + tripleMatch[0].length);
        if (end === -1) {
          tripleQuote = delimiter;
          add(rest, "syntax-string");
          return html;
        }
        add(line.slice(index, end + 3), "syntax-string");
        index = end + 3;
        continue;
      }

      if (line[index] === "#") {
        add(line.slice(index), "syntax-comment");
        break;
      }

      const stringMatch = rest.match(/^(?:[rRuUbBfF]{0,2})(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/);
      if (stringMatch) {
        add(stringMatch[0], "syntax-string");
        index += stringMatch[0].length;
        continue;
      }

      const decoratorMatch = rest.match(/^@[A-Za-z_][A-Za-z0-9_.]*/);
      if (decoratorMatch) {
        add(decoratorMatch[0], "syntax-decorator");
        index += decoratorMatch[0].length;
        continue;
      }

      const numberMatch = rest.match(/^(?:0[xob][0-9a-f_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:e[+-]?\d+)?)/i);
      if (numberMatch) {
        add(numberMatch[0], "syntax-number");
        index += numberMatch[0].length;
        continue;
      }

      const identifierMatch = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (identifierMatch) {
        const word = identifierMatch[0];
        let className = "";
        if (expectDefinition) {
          className = "syntax-definition";
          expectDefinition = false;
        } else if (keywords.has(word)) {
          className = "syntax-keyword";
          expectDefinition = word === "def" || word === "class";
        } else if (builtins.has(word)) {
          className = "syntax-builtin";
        }
        add(word, className);
        index += word.length;
        continue;
      }

      if (/^[+\-*/%=<>!&|^~:]+/.test(rest)) {
        const operator = rest.match(/^[+\-*/%=<>!&|^~:]+/)[0];
        add(operator, "syntax-operator");
        index += operator.length;
        continue;
      }

      add(line[index]);
      index += 1;
    }

    return html || " ";
  }

  function highlightMarkdownLine(line) {
    const escaped = escapeHtml(line);
    if (/^#{1,6}\s/.test(line)) {
      return `<span class="syntax-definition">${escaped}</span>`;
    }
    if (/^```/.test(line)) {
      return `<span class="syntax-decorator">${escaped}</span>`;
    }
    if (/^\s*(?:[-*+]|\d+\.)\s/.test(line)) {
      return escaped.replace(/^(\s*(?:[-*+]|\d+\.))/, '<span class="syntax-keyword">$1</span>');
    }
    if (/^\s*&gt;/.test(escaped)) {
      return `<span class="syntax-comment">${escaped}</span>`;
    }
    return escaped
      .replace(/(`[^`]+`)/g, '<span class="syntax-string">$1</span>')
      .replace(/(\*\*[^*]+\*\*)/g, '<span class="syntax-keyword">$1</span>')
      .replace(/(\[[^\]]+\]\([^)]+\))/g, '<span class="syntax-builtin">$1</span>');
  }

  function sourceRows(text, language) {
    tripleQuote = null;
    return text.replace(/\r\n/g, "\n").split("\n").map((line, lineIndex) => {
      const highlighted = language === "python"
        ? highlightPythonLine(line)
        : language === "markdown"
          ? highlightMarkdownLine(line)
          : escapeHtml(line) || " ";
      return `
        <span class="code-line">
          <span class="code-line-number" aria-hidden="true">${lineIndex + 1}</span>
          <span class="code-line-content">${highlighted}</span>
        </span>
      `;
    }).join("");
  }

  function resolveMarkdownUrl(url, rawBase, repoBase, isImage) {
    const normalized = url.trim().replace(/^<|>$/g, "");
    if (/^https?:\/\//i.test(normalized) || normalized.startsWith("#")) {
      return normalized;
    }
    if (/^(?:javascript|data):/i.test(normalized)) return "#";
    const path = normalized.replace(/^\.\//, "");
    return `${isImage ? rawBase : `${repoBase}/blob/main/`}${path}`;
  }

  function inlineMarkdown(value, rawBase, repoBase) {
    const normalizedImages = value.replace(
      /<img\s+[^>]*src=["']([^"']+)["'][^>]*alt=["']([^"']*)["'][^>]*>/gi,
      "![$2]($1)"
    );
    let html = escapeHtml(normalizedImages);
    html = html.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, alt, url) => {
      const resolved = resolveMarkdownUrl(url.replaceAll("&amp;", "&"), rawBase, repoBase, true);
      return `<img src="${escapeAttribute(resolved)}" alt="${escapeAttribute(alt)}" loading="lazy">`;
    });
    html = html.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, label, url) => {
      const resolved = resolveMarkdownUrl(url.replaceAll("&amp;", "&"), rawBase, repoBase, false);
      return `<a href="${escapeAttribute(resolved)}" target="_blank" rel="noopener">${label}</a>`;
    });
    return html
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
  }

  function markdownPreview(markdown, rawBase, repoBase) {
    const lines = markdown.replace(/\r\n/g, "\n").split("\n");
    const output = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      const trimmed = line.trim();

      if (!trimmed) {
        index += 1;
        continue;
      }

      const fence = trimmed.match(/^```(.*)$/);
      if (fence) {
        const language = fence[1].trim();
        const codeLines = [];
        index += 1;
        while (index < lines.length && !lines[index].trim().startsWith("```")) {
          codeLines.push(lines[index]);
          index += 1;
        }
        index += 1;
        output.push(`<pre><code data-language="${escapeAttribute(language)}">${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        continue;
      }

      const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        const level = heading[1].length;
        output.push(`<h${level}>${inlineMarkdown(heading[2], rawBase, repoBase)}</h${level}>`);
        index += 1;
        continue;
      }

      if (/^(?:---+|\*\*\*+|___+)$/.test(trimmed)) {
        output.push("<hr>");
        index += 1;
        continue;
      }

      if (line.includes("|") && index + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[index + 1])) {
        const rows = [line];
        index += 2;
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          rows.push(lines[index]);
          index += 1;
        }
        const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|");
        const headers = cells(rows[0]);
        output.push("<div class=\"markdown-table-wrap\"><table><thead><tr>");
        headers.forEach((cell) => output.push(`<th>${inlineMarkdown(cell.trim(), rawBase, repoBase)}</th>`));
        output.push("</tr></thead><tbody>");
        rows.slice(1).forEach((row) => {
          output.push("<tr>");
          cells(row).forEach((cell) => output.push(`<td>${inlineMarkdown(cell.trim(), rawBase, repoBase)}</td>`));
          output.push("</tr>");
        });
        output.push("</tbody></table></div>");
        continue;
      }

      if (/^\s*[-*+]\s+/.test(line) || /^\s*\d+\.\s+/.test(line)) {
        const ordered = /^\s*\d+\.\s+/.test(line);
        const tag = ordered ? "ol" : "ul";
        output.push(`<${tag}>`);
        while (index < lines.length) {
          const match = lines[index].match(ordered ? /^\s*\d+\.\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/);
          if (!match) break;
          output.push(`<li>${inlineMarkdown(match[1], rawBase, repoBase)}</li>`);
          index += 1;
        }
        output.push(`</${tag}>`);
        continue;
      }

      if (/^\s*>\s?/.test(line)) {
        const quote = [];
        while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
          quote.push(lines[index].replace(/^\s*>\s?/, ""));
          index += 1;
        }
        output.push(`<blockquote>${quote.map((item) => inlineMarkdown(item, rawBase, repoBase)).join("<br>")}</blockquote>`);
        continue;
      }

      if (/^\s*<\/?(?:div|p|a)(?:\s|>)/i.test(line)) {
        index += 1;
        continue;
      }

      output.push(`<p>${inlineMarkdown(trimmed, rawBase, repoBase)}</p>`);
      index += 1;
    }

    return output.join("");
  }

  function setLoading(message) {
    body.innerHTML = `<div class="code-viewer-status" role="status">${message}</div>`;
    copyButton.disabled = true;
    activeText = "";
  }

  function setError(message) {
    body.innerHTML = `<div class="code-viewer-status" data-state="error" role="status">${message}</div>`;
    copyButton.disabled = true;
  }

  async function fetchText(url) {
    if (!cache.has(url)) {
      const request = fetch(url)
        .then((response) => {
          if (!response.ok) throw new Error(`Request failed (${response.status})`);
          return response.text();
        })
        .catch((error) => {
          cache.delete(url);
          throw error;
        });
      cache.set(url, request);
    }
    return cache.get(url);
  }

  async function fetchJson(url) {
    if (!cache.has(url)) {
      const request = fetch(url)
        .then((response) => {
          if (!response.ok) throw new Error(`Request failed (${response.status})`);
          return response.json();
        })
        .catch((error) => {
          cache.delete(url);
          throw error;
        });
      cache.set(url, request);
    }
    return cache.get(url);
  }

  function renderText(text, options, mode = options.defaultMode || "source") {
    activeText = text;
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const canPreview = options.language === "markdown";
    title.textContent = options.title;
    meta.textContent = `${options.label} · ${lines.length} lines`;
    const fileBody = canPreview && mode === "preview"
      ? `<article class="markdown-preview">${markdownPreview(text, options.rawBase, options.repoBase)}</article>`
      : `
        <div class="code-viewer-scroll">
          <code class="code-viewer-code" aria-label="${options.ariaLabel}">
            ${sourceRows(text, options.language)}
          </code>
        </div>
      `;

    body.innerHTML = canPreview ? `
      <section class="repo-file repo-file--standalone" aria-label="${escapeAttribute(options.title)}">
        <div class="repo-file-heading">
          <div class="repo-file-path" title="${escapeAttribute(options.title)}">
            <strong>${escapeHtml(options.title)}</strong>
          </div>
          <div class="repo-view-toggle" aria-label="Markdown display mode">
            <button type="button" data-viewer-mode="preview" class="${mode === "preview" ? "is-active" : ""}">Preview</button>
            <button type="button" data-viewer-mode="source" class="${mode === "source" ? "is-active" : ""}">Source</button>
          </div>
        </div>
        <div class="repo-file-content">${fileBody}</div>
      </section>
    ` : fileBody;

    body.querySelectorAll("[data-viewer-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        renderText(text, options, button.dataset.viewerMode);
      });
    });
    copyButton.disabled = false;
  }

  function buildRepositoryTree(items) {
    const root = { children: new Map() };
    items
      .filter((item) => item.type === "tree" || item.type === "blob")
      .sort((a, b) => a.path.localeCompare(b.path))
      .forEach((item) => {
        let parent = root;
        const parts = item.path.split("/");
        parts.forEach((part, partIndex) => {
          if (!parent.children.has(part)) {
            parent.children.set(part, {
              name: part,
              path: parts.slice(0, partIndex + 1).join("/"),
              type: partIndex === parts.length - 1 ? item.type : "tree",
              children: new Map()
            });
          }
          const node = parent.children.get(part);
          if (partIndex === parts.length - 1) node.type = item.type;
          parent = node;
        });
      });
    return root;
  }

  function repositoryTreeMarkup(node, depth = 0) {
    const children = [...node.children.values()].sort((a, b) => {
      if (a.type !== b.type) return a.type === "tree" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    return `
      <ul class="repo-tree-list" role="${depth === 0 ? "tree" : "group"}">
        ${children.map((child) => {
          if (child.type === "tree") {
            return `
              <li class="repo-tree-node repo-tree-node--folder" role="treeitem">
                <details>
                  <summary>
                    <span class="repo-tree-chevron" aria-hidden="true"></span>
                    <span class="repo-tree-icon repo-tree-icon--folder" aria-hidden="true"></span>
                    <span>${escapeHtml(child.name)}</span>
                  </summary>
                  ${repositoryTreeMarkup(child, depth + 1)}
                </details>
              </li>
            `;
          }
          return `
            <li class="repo-tree-node repo-tree-node--file" role="treeitem">
              <button type="button" data-repo-file="${escapeAttribute(child.path)}">
                <span class="repo-tree-icon repo-tree-icon--file" aria-hidden="true"></span>
                <span>${escapeHtml(child.name)}</span>
              </button>
            </li>
          `;
        }).join("")}
      </ul>
    `;
  }

  function repositoryRawUrl(path) {
    return repositoryState.rawBase + path.split("/").map(encodeURIComponent).join("/");
  }

  function repositoryDisplayName() {
    return repositoryState?.displayName || "ghost-font-decoder";
  }

  function repositoryGithubUrl(path) {
    return `${repositoryState.repoBase}/blob/main/${path.split("/").map(encodeURIComponent).join("/")}`;
  }

  function repositoryFileKind(path) {
    if (/\.(?:png|jpe?g|gif|webp|svg|ico)$/i.test(path)) return "image";
    if (/\.(?:mp4|webm|mov)$/i.test(path)) return "video";
    if (/\.(?:zip|gz|woff2?|ttf|otf|pdf)$/i.test(path)) return "binary";
    return "text";
  }

  function setSelectedRepositoryFile(path) {
    body.querySelectorAll("[data-repo-file].is-selected").forEach((button) => {
      button.classList.remove("is-selected");
    });
    const selected = [...body.querySelectorAll("[data-repo-file]")]
      .find((button) => button.dataset.repoFile === path);
    selected?.classList.add("is-selected");
  }

  function renderRepositoryFile(path, text, mode) {
    const panel = body.querySelector(".repo-file");
    if (!panel) return;
    const isMarkdown = /\.md$/i.test(path);
    const filename = path.split("/").pop();
    repositoryState.currentPath = path;
    repositoryState.currentText = text;
    repositoryState.mode = isMarkdown ? mode : "source";
    activeText = text;
    copyButton.disabled = false;
    githubLink.href = repositoryGithubUrl(path);
    setSelectedRepositoryFile(path);

    const fileBody = isMarkdown && mode === "preview"
      ? `<article class="markdown-preview">${markdownPreview(text, repositoryState.rawBase, repositoryState.repoBase)}</article>`
      : `<div class="code-viewer-scroll"><code class="code-viewer-code" aria-label="${escapeAttribute(filename)} source">${sourceRows(text, /\.py$/i.test(path) ? "python" : /\.md$/i.test(path) ? "markdown" : "text")}</code></div>`;

    panel.innerHTML = `
      <div class="repo-file-heading">
        <div class="repo-file-path" title="${escapeAttribute(path)}">
          <span>${escapeHtml(repositoryDisplayName())}</span><span>/</span><strong>${escapeHtml(path)}</strong>
        </div>
        ${isMarkdown ? `
          <div class="repo-view-toggle" aria-label="Markdown display mode">
            <button type="button" data-repo-mode="preview" class="${mode === "preview" ? "is-active" : ""}">Preview</button>
            <button type="button" data-repo-mode="source" class="${mode === "source" ? "is-active" : ""}">Source</button>
          </div>
        ` : `<span class="repo-file-lines">${text.replace(/\r\n/g, "\n").split("\n").length} lines</span>`}
      </div>
      <div class="repo-file-content">${fileBody}</div>
    `;

    panel.querySelectorAll("[data-repo-mode]").forEach((button) => {
      button.addEventListener("click", () => {
        renderRepositoryFile(path, text, button.dataset.repoMode);
      });
    });
  }

  async function openRepositoryFile(path, prefetchedText) {
    const panel = body.querySelector(".repo-file");
    if (!panel || !repositoryState) return;
    const selectionId = ++repositoryState.selectionSequence;
    const kind = repositoryFileKind(path);
    const filename = path.split("/").pop();
    setSelectedRepositoryFile(path);
    githubLink.href = repositoryGithubUrl(path);
    panel.innerHTML = `
      <div class="repo-file-heading">
        <div class="repo-file-path"><span>${escapeHtml(repositoryDisplayName())}</span><span>/</span><strong>${escapeHtml(path)}</strong></div>
      </div>
      <div class="repo-file-loading" role="status">Loading ${escapeHtml(filename)}&hellip;</div>
    `;

    if (kind === "image" || kind === "video") {
      activeText = "";
      copyButton.disabled = true;
      const url = repositoryRawUrl(path);
      panel.innerHTML = `
        <div class="repo-file-heading">
          <div class="repo-file-path"><span>${escapeHtml(repositoryDisplayName())}</span><span>/</span><strong>${escapeHtml(path)}</strong></div>
          <span class="repo-file-lines">${kind} preview</span>
        </div>
        <div class="repo-media-preview">
          ${kind === "image"
            ? `<img src="${escapeAttribute(url)}" alt="Preview of ${escapeAttribute(filename)}">`
            : `<video src="${escapeAttribute(url)}" controls preload="metadata"></video>`}
        </div>
      `;
      return;
    }

    if (kind === "binary") {
      activeText = "";
      copyButton.disabled = true;
      panel.innerHTML = `
        <div class="repo-file-heading">
          <div class="repo-file-path"><span>${escapeHtml(repositoryDisplayName())}</span><span>/</span><strong>${escapeHtml(path)}</strong></div>
        </div>
        <div class="repo-file-empty">This binary file cannot be previewed here. Use the GitHub button to open it.</div>
      `;
      return;
    }

    try {
      const text = prefetchedText ?? await fetchText(repositoryRawUrl(path));
      if (!repositoryState || selectionId !== repositoryState.selectionSequence) return;
      renderRepositoryFile(path, text, /\.md$/i.test(path) ? "preview" : "source");
    } catch {
      if (!repositoryState || selectionId !== repositoryState.selectionSequence) return;
      panel.innerHTML = `<div class="repo-file-empty" data-state="error">${escapeHtml(filename)} could not be loaded. Use the GitHub button to open it.</div>`;
    }
  }

  async function openRepository(trigger, requestId) {
    const viewerTitle = trigger.dataset.viewerTitle || "ghost-font-decoder";
    const initialFile = trigger.dataset.repoInitialFile || "README.md";
    const initialFileUrl = trigger.dataset.repoInitialFileViewer || trigger.dataset.readmeViewer;
    title.textContent = viewerTitle;
    meta.textContent = "Repository source";
    setLoading("Loading repository files&hellip;");

    const [treeResult, initialFileResult] = await Promise.allSettled([
      fetchJson(trigger.dataset.repoViewer),
      fetchText(initialFileUrl)
    ]);
    if (requestId !== requestSequence) return;

    if (treeResult.status === "rejected" && initialFileResult.status === "rejected") {
      setError("The repository could not be loaded. Use the GitHub button to open the source.");
      return;
    }

    const tree = treeResult.status === "fulfilled" ? treeResult.value.tree || [] : [];
    const roots = (trigger.dataset.repoRoots || "")
      .split(",")
      .map((root) => root.trim().replace(/\/$/, ""))
      .filter(Boolean);
    const visibleTree = tree.filter((item) => {
      if (roots.length) {
        return roots.some((root) => item.path === root || item.path.startsWith(`${root}/`));
      }
      return item.path !== "docs" && !item.path.startsWith("docs/");
    });
    const initialText = initialFileResult.status === "fulfilled"
      ? initialFileResult.value
      : `# ${initialFile} unavailable\n\nUse the GitHub button to view this file.`;
    const fileCount = visibleTree.filter((item) => item.type === "blob").length;
    const folderCount = visibleTree.filter((item) => item.type === "tree").length;
    const rawBase = trigger.dataset.readmeViewer.slice(0, trigger.dataset.readmeViewer.lastIndexOf("/") + 1);
    repositoryState = {
      tree: visibleTree,
      rawBase,
      repoBase: trigger.href.replace(/\/$/, ""),
      displayName: viewerTitle,
      selectionSequence: 0,
      currentPath: "",
      currentText: "",
      mode: "preview"
    };
    meta.textContent = `${folderCount} folders · ${fileCount} files`;
    body.innerHTML = `
      <div class="repo-viewer">
        <aside class="repo-tree" aria-label="Repository folders and files">
          <div class="repo-tree-heading">
            <span>Files</span>
            <span>${fileCount} files</span>
          </div>
          ${visibleTree.length
            ? repositoryTreeMarkup(buildRepositoryTree(visibleTree))
            : '<p class="repo-tree-error">Tree unavailable. Open GitHub for the full file list.</p>'}
        </aside>
        <section class="repo-file" aria-label="Selected repository file"></section>
      </div>
    `;
    body.querySelectorAll("[data-repo-file]").forEach((button) => {
      button.addEventListener("click", () => openRepositoryFile(button.dataset.repoFile));
    });
    await openRepositoryFile(initialFile, initialText);
  }

  async function openTrigger(trigger) {
    const requestId = ++requestSequence;
    activeTrigger = trigger;
    githubLink.href = trigger.href;
    dialog.showModal();

    if (trigger.dataset.repoViewer) {
      await openRepository(trigger, requestId);
      return;
    }

    const isPython = Boolean(trigger.dataset.codeViewer);
    const url = trigger.dataset.codeViewer || trigger.dataset.textViewer;
    const filename = trigger.dataset.viewerTitle || url.split("/").pop() || (isPython ? "decode.py" : "File");
    const language = trigger.dataset.viewerLanguage || (isPython ? "python" : "text");
    const label = trigger.dataset.viewerLabel || (isPython ? "Python source" : "Plain text");
    const ariaLabel = trigger.dataset.viewerAriaLabel || `${filename} source`;
    const rawMarker = "/main/";
    const rawBase = url.includes(rawMarker)
      ? url.slice(0, url.indexOf(rawMarker) + rawMarker.length)
      : url.slice(0, url.lastIndexOf("/") + 1);
    const repoBase = trigger.href.includes("/blob/")
      ? trigger.href.slice(0, trigger.href.indexOf("/blob/"))
      : trigger.href.replace(/\/$/, "");
    title.textContent = filename;
    meta.textContent = label;
    setLoading(`Loading ${escapeHtml(filename)}&hellip;`);

    try {
      const text = await fetchText(url);
      if (requestId !== requestSequence) return;
      renderText(text, {
        title: filename,
        label,
        language,
        ariaLabel,
        rawBase,
        repoBase,
        defaultMode: trigger.dataset.viewerDefaultMode || "source"
      });
    } catch {
      if (requestId !== requestSequence) return;
      setError(`${escapeHtml(filename)} could not be loaded. Use the GitHub button to open it.`);
    }
  }

  triggers.forEach((trigger) => {
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      openTrigger(trigger);
    });
  });

  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => activeTrigger?.focus());

  copyButton.addEventListener("click", async () => {
    if (!activeText) return;
    try {
      let copyText = activeText;
      const heading = activeTrigger?.dataset.viewerCopyAfterHeading;
      if (heading) {
        const lines = copyText.replace(/\r\n/g, "\n").split("\n");
        const headingIndex = lines.findIndex((line) => line.trim() === heading);
        if (headingIndex === -1) throw new Error("Copy boundary not found");
        copyText = lines.slice(headingIndex + 1).join("\n").trim();
      }
      await navigator.clipboard.writeText(copyText);
      copyButton.textContent = "Copied";
    } catch {
      copyButton.textContent = "Copy failed";
    }
    window.setTimeout(() => { copyButton.textContent = "Copy"; }, 1600);
  });
})();
