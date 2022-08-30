const readline = require("readline")
const fs = require("fs")
const path = require("path")

const prefix = "data/formatted"

function processLine(line) {
  const { id, articles, summary } = JSON.parse(line)

  console.log("processing", id)

  fs.mkdirSync(path.join(prefix, String(id)))

  fs.writeFileSync(path.join(prefix, String(id), "summary.txt"), summary)

  fs.mkdirSync(path.join(prefix, String(id), "documents"))

  articles.forEach(({ text }) =>
    fs.writeFileSync(
      path.join(prefix, String(id), "documents.txt"),
      text + "\n\n",
      { flag: "a+" }
    )
  )
}

try {
  fs.rmdirSync(prefix, { recursive: true })
} catch (error) {
  console.log(error)
  // Allowed to fail if no directory exists
}

fs.mkdirSync(prefix)

const readInterface = readline.createInterface({
  input: fs.createReadStream("data/test.jsonl"),
  output: false,
})

readInterface.on("line", processLine)

readInterface.on("close", function () {
  console.log("done")
})
