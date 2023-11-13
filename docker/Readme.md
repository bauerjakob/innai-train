# InnAi Docker

# How to add user to Database

1. Terminal in Mongo Container
2. Open Mongo-Shell `mongosh`
3. Create User
```
use InnAi
// db.auth("root", "password")
db.createUser(
  {
    user: "jakob",
    pwd: "password",
    roles: [ { role: "dbOwner", db: "InnAi" }, { role: "readWrite", db: "InnAi" } ]
  }
)
```