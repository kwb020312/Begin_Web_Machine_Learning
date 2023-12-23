const express = require("express");
const app = express();

app.use(express.static("www"));

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/www/index.html");
});

const listener = app.listen(3000, () => {
  console.log("Your app is Listening On Port 3000");
});
