const readline = require("readline")
const fs = require("fs")
const path = require("path")

const prefix = process.argv[3] ?? "data/formatted"

const articleNumbers = []

const filename = process.argv[2] ?? "data/t.jsonl"
const datasetName = path.basename(filename).split(".").slice(0, -1).join(".")

function processLine(line) {
  const { id, articles, summary } = JSON.parse(line)

  console.log("processing", id)

  const dirPath = path.join(prefix, datasetName + String(id))

  fs.mkdirSync(dirPath)

  fs.writeFileSync(path.join(dirPath, "summary.txt"), summary)

  fs.writeFileSync(
    path.join(dirPath, "documents.txt"),
    articles.map((article) => article.text.replace(/\n\n/g, "\n")).join("\n\n")
  )

  articleNumbers.push(articles.length)
}

try {
  fs.rmdirSync(prefix, { recursive: true })
} catch (error) {
  console.log(error)
  // Allowed to fail if no directory exists
}

fs.mkdirSync(prefix, { recursive: true })

const readInterface = readline.createInterface({
  input: fs.createReadStream(filename),
  output: false,
})

readInterface.on("line", processLine)

readInterface.on("close", function () {
  console.log(
    "avg Articles per Summary",
    articleNumbers.reduce((prev, curr) => prev + curr, 0) /
      articleNumbers.length
  )
  console.log("done")
})
