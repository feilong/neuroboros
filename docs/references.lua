-- Reference list markup for the neuroboros documentation.
--
-- Runs citeproc (the project sets `citeproc: false` so that it is not run
-- before this filter) and then decorates each entry that neuroboros.csl
-- produced, using references-meta.json:
--
--   * the author block: each name becomes a <span class="au ...">, bold for
--     first, senior, and corresponding authors, with a native tooltip naming
--     the roles, and an envelope icon after each corresponding author,
--     inside the span so that it stays on the same line as the name;
--   * the venue block: a paper-type icon (empirical, review, data) in front,
--     and a "URL" link to the paper (the title's link) at the end.
--
-- The classes and icons follow the lab website's publication listing, so
-- styles.css carries the same rules. Entries without metadata only get the
-- URL link.

local CATEGORY = {
  empirical = { icon = "fa-flask", label = "Empirical" },
  review = { icon = "fa-book-open", label = "Review / Perspective" },
  data = { icon = "fa-database", label = "Data / Resources" },
}
local ROLE_LABEL = {
  first = "First author",
  senior = "Senior author",
  corresponding = "Corresponding author",
}

local function warn(msg)
  io.stderr:write("references.lua: " .. msg .. "\n")
end

local function load_meta()
  local dir = quarto and quarto.project and quarto.project.directory or "."
  local fn = pandoc.path.join({ dir, "references-meta.json" })
  local f = io.open(fn, "r")
  if not f then
    warn("cannot read " .. fn)
    return {}
  end
  local text = f:read("a")
  f:close()
  return pandoc.json.decode(text)
end

local function escape(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;"))
end

local ENVELOPE = pandoc.RawInline("html",
  '<sup class="au-corr" title="corresponding author">'
  .. '<i class="fa-solid fa-envelope" aria-hidden="true"></i>'
  .. '<span class="visually-hidden"> (corresponding author)</span></sup>')

-- Split the author block, "A, B, C" as neuroboros.csl writes it, into names.
local function split_names(inlines)
  local names = {}
  for name in pandoc.utils.stringify(inlines):gmatch("[^,]+") do
    names[#names + 1] = name:gsub("^%s+", ""):gsub("%s+$", "")
  end
  return names
end

local function author_inlines(names, roles)
  local out = pandoc.Inlines({})
  for i, name in ipairs(names) do
    local classes, labels, corr = { "au" }, {}, false
    for role in (roles[i] or ""):gmatch("%S+") do
      if role == "first" then classes[#classes + 1] = "au-first" end
      if role == "senior" then classes[#classes + 1] = "au-senior" end
      if role == "corresponding" then
        classes[#classes + 1] = "au-corr-name"
        corr = true
      end
      labels[#labels + 1] = ROLE_LABEL[role] or role
    end
    local attr = pandoc.Attr("", classes,
      #labels > 0 and { title = table.concat(labels, " · ") } or {})
    if i > 1 then
      out:insert(pandoc.Str(","))
      out:insert(pandoc.Space())
    end
    -- the envelope goes inside the name's span, which does not wrap, so the
    -- two never land on different lines
    local content = pandoc.Inlines({ pandoc.Str(name) })
    if corr then content:insert(ENVELOPE) end
    out:insert(pandoc.Span(content, attr))
  end
  return out
end

local function category_inline(kind)
  local cat = CATEGORY[kind]
  if not cat then
    warn("unknown paper type '" .. tostring(kind) .. "'")
    return nil
  end
  return pandoc.RawInline("html",
    '<i class="fa-solid ' .. cat.icon .. ' pub-cat pub-cat-' .. kind
    .. '" data-tip="' .. escape(cat.label) .. ' paper" aria-hidden="true"></i>'
    .. '<span class="visually-hidden">' .. escape(cat.label) .. ' paper. </span>')
end

-- citeproc renders each display="block" group as a Span (or Div) with the
-- class csl-block; neuroboros.csl emits title, authors, venue in that order.
local function decorate(entry, meta)
  local n, url = 0, nil
  return entry:walk({
    Span = function(el)
      if not el.classes:includes("csl-block") then return nil end
      n = n + 1
      if n == 1 then
        -- citeproc links the title to the DOI (or URL) of the entry
        el:walk({ Link = function(link) url = url or link.target end })
      elseif n == 2 and meta.authors then
        local names = split_names(el.content)
        if #names ~= #meta.authors then
          warn(string.format("%s: %d authors in the bibliography, %d in references-meta.json",
            entry.identifier, #names, #meta.authors))
          return nil
        end
        el.content = author_inlines(names, meta.authors)
        return el
      elseif n == 3 then
        local icon = meta.type and category_inline(meta.type)
        if icon then el.content:insert(1, icon) end
        if url then
          el.content:insert(pandoc.Space())
          el.content:insert(pandoc.Link("URL", url, "", pandoc.Attr("", { "pub-link" })))
        end
        return el
      end
    end,
  })
end

function Pandoc(doc)
  doc = pandoc.utils.citeproc(doc)
  local meta = load_meta()
  return doc:walk({
    Div = function(el)
      if not el.classes:includes("csl-entry") then return nil end
      local key = el.identifier:gsub("^ref%-", "")
      return decorate(el, meta[key] or {})
    end,
  })
end
