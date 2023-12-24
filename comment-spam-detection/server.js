const { createServer } = require("http");
const express = require("express");
const { Server } = require("socket.io");

const app = express();
const server = createServer(app);
const io = new Server(server);

app.use(express.static("www"));

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/www/index.html");
});

io.on("connect", (socket) => {
  console.log("Client Connected");
  socket.on("comment", (data) => {
    socket.broadcast.emit("remoteComment", data);
  });
});

const listener = app.listen(3000, () => {
  console.log("Your app is Listening On Port 3000");
});
