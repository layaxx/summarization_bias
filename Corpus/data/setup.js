const readline = require("readline")
const fs = require("fs")
const path = require("path")

const prefix = process.argv[3] ?? "data/formatted"

const articleNumbers = []

function processLine(line) {
  const { id, articles, summary } = JSON.parse(line)

  console.log("processing", id)

  fs.mkdirSync(path.join(prefix, String(id)))

  fs.writeFileSync(path.join(prefix, String(id), "summary.txt"), summary)

  articles.forEach(({ text }) =>
    fs.writeFileSync(
      path.join(prefix, String(id), "documents.txt"),
      text + "\n\n",
      { flag: "a+" }
    )
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
  input: fs.createReadStream(process.argv[2] ?? "data/t.jsonl"),
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
