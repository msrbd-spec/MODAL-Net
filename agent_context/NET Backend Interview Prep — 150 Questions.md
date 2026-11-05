# .NET Backend Interview Prep — 150 Questions

Nov 4, 2026 · @Hasan

## How to use this guide

Questions 1–50 follow the job description's weighting: C# and OOP fundamentals, ASP.NET Core and REST APIs, and SQL carry most of the weight; EF Core, Git, Redis, AWS and Jenkins come up as "nice to have" checks; Q44–50 are scenario questions. Each answer has a **one-line version** (say this first) and a **descriptive version** (expand if the interviewer lets you keep talking).

Tips for a 30-minute AI interview:

- **Lead with the one-liner, then add one example.** AI interviewers usually score on correctness, clarity and structure; a crisp definition plus a concrete example beats a long ramble.
- **Say "I don't know, but here's how I'd find out" when stuck.** The JD explicitly values willingness to learn and ask questions.
- **Tie answers to your project.** Have one .NET project ready to describe in 60 seconds: what it does, the stack, one bug you fixed, one thing you'd improve. Keep the GitHub link handy.
- **For scenario questions, use a structure:** clarify → investigate (logs, reproduce) → fix → test → prevent recurrence.
- **Check your setup:** camera, mic, quiet room, stable internet, and speak at a steady pace.

## OOP & C# fundamentals (Q1–10)

### Q1. What are the four pillars of OOP?

**One line:** Encapsulation, Abstraction, Inheritance and Polymorphism.

**Descriptive:**

- **Encapsulation** — keep the data and the methods that use it together in one class, and hide the data from outside. Example: a `BankAccount` class where `Balance` can't be changed directly; you must call `Deposit()`.
- **Abstraction** — show *what* an object does, hide *how* it does it. Interfaces and abstract classes help with this.
- **Inheritance** — one class can reuse and extend another class. Example: `SavingsAccount : BankAccount`.
- **Polymorphism** — the same method call can behave differently depending on the object. Example: `shape.Area()` works differently for a circle and a square.

### Q2. Abstract class vs interface — when do you use each?

**One line:** An interface is only a contract ("this class *can do* this"). An abstract class is a base class with some shared code ("this class *is a* kind of that").

**Descriptive:** A class can implement many interfaces, but it can inherit only one class. An abstract class can have fields, constructors and normal methods with code. An interface mainly lists methods that a class must have. Use an interface when different, unrelated classes need the same ability, and for dependency injection (for example `IEmailService`). Use an abstract class when related classes share real code (for example a `Shape` class with a shared `Describe()` method and an abstract `Area()` method).

### Q3. What's the difference between value types and reference types?

**One line:** A value type stores the actual value. A reference type stores an address (reference) that points to the object in memory.

**Descriptive:** Value types: `int`, `double`, `bool`, `DateTime`, `enum`, `struct`. Reference types: `class`, `string`, arrays. When you copy a value type, you get a separate copy — changing one does not change the other. When you copy a reference type, both variables point to the *same* object — changing one changes what both see. Putting a value type into an `object` is called *boxing*; it is slower, so avoid it in loops. `string` is a reference type, but it acts like a value because it can't be changed.

### Q4. Method overloading vs overriding?

**One line:** Overloading = same method name with different parameters. Overriding = a child class gives its own version of a parent's method.

**Descriptive:** Overloading example: `Add(int a, int b)` and `Add(double a, double b)` in the same class. The compiler picks the right one. Overriding example: the parent marks a method `virtual` (or `abstract`), and the child uses `override`. The right version is chosen at runtime based on the real object. If you use `new` instead of `override`, you only *hide* the parent's method — this often causes bugs. `sealed` stops further overriding.

### Q5. Explain C# access modifiers.

**One line:** They control who can use a class or member: `public`, `private`, `protected`, `internal`, `protected internal`, `private protected`.

**Descriptive:**

- `public` — anyone can use it.
- `private` — only the same class (default for class members).
- `protected` — the same class and its child classes.
- `internal` — any code in the same project (assembly). Default for classes.
- `protected internal` — same project **or** child classes.
- `private protected` — child classes, but only inside the same project.

Good habit: keep fields `private` and give access through properties.

### Q6. Why is `string` immutable, and when should you use `StringBuilder`?

**One line:** A string can't be changed — every change makes a new string. So use `StringBuilder` when you add to a string many times, like in a loop.

**Descriptive:** If you write `s += x` inside a loop 10,000 times, .NET creates 10,000 new strings. This wastes memory and time. `StringBuilder` keeps one growing buffer and makes the final string once with `ToString()`. For just a few joins, normal `+` or `$"Hi {name}"` is fine and easier to read. Strings being unchangeable also makes them safe to share between threads.

### Q7. How do `async` and `await` work? Why do they matter in a web API?

**One line:** They let the server wait for slow work (database, HTTP calls, files) without blocking a thread, so it can serve more users at the same time.

**Descriptive:** An `async` method returns a `Task` or `Task<T>`. When the code hits `await` on slow work, the thread is freed to handle other requests. When the work finishes, the method continues. Rules to remember:

- Use async all the way, e.g. `await _db.Users.ToListAsync()`.
- Don't use `.Result` or `.Wait()` — they block and can cause deadlocks.
- Avoid `async void` (only for event handlers).
- Pass a `CancellationToken` when you can.

Async doesn't make one request faster. It helps the server handle *more* requests.

### Q8. How do you handle exceptions in C#? `throw` vs `throw ex`?

**One line:** Use `try/catch/finally`, catch specific exceptions, and use `throw;` (not `throw ex;`) to rethrow so you keep the original error location.

**Descriptive:** Catch specific errors like `SqlException`, not every `Exception`, unless it's a global handler. `finally` (or `using`) always runs, so use it for cleanup like closing a connection. `throw ex;` loses where the error first happened; `throw;` keeps it. Don't use exceptions for normal checks — for example, use `int.TryParse` instead of catching `FormatException`. In ASP.NET Core, one global error-handling middleware can log all errors and return a clean error response.

### Q9. `IEnumerable` vs `IQueryable` vs `List` — and what is LINQ?

**One line:** LINQ lets you query data in C#. `IQueryable` runs the filter in the database, `IEnumerable` filters in memory, and `List` is data already loaded in memory.

**Descriptive:** With EF Core, `_db.Users.Where(u => u.IsActive)` becomes a SQL `WHERE`, so only the needed rows come from the database. But if you call `.ToList()` first and then filter, *all* rows are loaded into memory first — a common performance mistake. LINQ queries run only when you actually use the result (`foreach`, `ToList()`, `Count()`, `First()`). This is called *deferred execution*. Common methods: `Where`, `Select`, `OrderBy`, `GroupBy`, `Any`, `FirstOrDefault`.

### Q10. What are the SOLID principles?

**One line:** Five rules for writing clean, easy-to-change code.

**Descriptive:**

- **S – Single Responsibility:** one class should do one job. A controller shouldn't also send emails and write SQL.
- **O – Open/Closed:** add new features by adding new code, not by changing old working code. Example: add a new `IPaymentProcessor` class instead of editing a big `switch`.
- **L – Liskov Substitution:** a child class should work anywhere the parent class works, without breaking anything.
- **I – Interface Segregation:** use small interfaces, so classes don't have to implement methods they don't need.
- **D – Dependency Inversion:** depend on interfaces, not on concrete classes. ASP.NET Core's dependency injection helps with this.

## .NET & ASP.NET Core (Q11–18)

### Q11. What is .NET? How do the CLR, JIT and garbage collector fit in?

**One line:** .NET is a platform for building and running apps. C# code is turned into IL, the CLR runs it, the JIT turns it into machine code, and the garbage collector (GC) cleans up unused memory.

**Descriptive:** Modern .NET (like .NET 8 or 9) runs on Windows, Linux and Mac and is open source. The old .NET Framework only ran on Windows. ".NET Core" was the earlier name of modern .NET. The steps:

1. The C# compiler turns your code into *IL* (Intermediate Language) inside a DLL.
2. The *CLR* (Common Language Runtime) loads and runs it.
3. The *JIT* (Just-In-Time) compiler turns IL into real machine code while the app runs.
4. The *GC* (garbage collector) finds objects nobody uses anymore and frees their memory.

The GC doesn't quickly close things like database connections or files. That's why those classes implement `IDisposable`, and you wrap them in `using`.

### Q12. What is middleware in ASP.NET Core, and why does order matter?

**One line:** Middleware are small steps that every request passes through, one after another. They run in the order you add them.

**Descriptive:** In `Program.cs`, lines like `app.UseExceptionHandler()`, `app.UseHttpsRedirection()`, `app.UseRouting()`, `app.UseAuthentication()`, `app.UseAuthorization()` and `app.MapControllers()` build this chain. The request goes down the chain, and the response comes back up. Order matters. For example, authentication (who is the user?) must come before authorization (is the user allowed?). The error handler goes first so it can catch errors from all later steps. You can also write your own middleware, for example to log every request or measure how long it takes.

### Q13. What is dependency injection? Explain Transient, Scoped and Singleton.

**One line:** Dependency injection (DI) means a class gets the objects it needs from outside (usually through its constructor) instead of creating them itself. The lifetime decides how long each object lives.

**Descriptive:** You register services in `Program.cs`, e.g. `builder.Services.AddScoped<IOrderService, OrderService>()`. ASP.NET Core then passes them into constructors for you. This keeps classes loosely connected and makes unit testing easy, because you can pass a fake (mock) object instead.

- **Transient** — a new object every time someone asks for it. Good for small, simple services.
- **Scoped** — one object per HTTP request. Used for `DbContext` and most business services.
- **Singleton** — one object for the whole app, shared by everyone. Good for caches, settings and Redis connections. It must be thread-safe.

Common mistake: putting a Scoped service (like `DbContext`) inside a Singleton. Then one request's `DbContext` gets shared by all requests, which causes bugs. This is called a *captive dependency*.

### Q14. Controllers vs Minimal APIs — and what does `[ApiController]` do?

**One line:** Controllers put related endpoints inside a class. Minimal APIs write endpoints directly in `Program.cs` with short code.

**Descriptive:** Controllers use attributes like `[Route("api/[controller]")]` and `[HttpGet("{id}")]`. They're good for bigger apps with many endpoints. Minimal APIs (`app.MapGet("/users/{id}", ...)`) are shorter and good for small services. `[ApiController]` gives helpful API features:

- It returns 400 Bad Request automatically when input is invalid.
- It guesses where data comes from: `[FromBody]` for objects, `[FromRoute]`/`[FromQuery]` for simple values.
- It requires attribute routing.

Return `ActionResult<T>` so you can return either data or a status like `NotFound()`.

### Q15. How do you validate incoming request data?

**One line:** Put validation attributes like `[Required]`, `[StringLength]`, `[Range]` and `[EmailAddress]` on your request classes (DTOs). With `[ApiController]`, bad input automatically gets a 400 response.

**Descriptive:** Use separate request classes (DTOs), not your database entity classes. Otherwise a user could send extra fields like `IsAdmin` and change data they shouldn't (*over-posting*). For more complex rules, many teams use the FluentValidation library. Rules that need the database, like "this email is already used", go in the service layer and return 409 Conflict or 400 with a clear message. Never trust user input — and always use parameterized queries to stop SQL injection.

### Q16. How does configuration work in ASP.NET Core? Where do you keep secrets?

**One line:** Settings come from `appsettings.json`, then `appsettings.{Environment}.json`, then environment variables and others. Later sources override earlier ones.

**Descriptive:** The `ASPNETCORE_ENVIRONMENT` value (Development, Staging, Production) decides which extra settings file is used. You can read values with `IConfiguration`. A cleaner way is the Options pattern: load a settings section into a class, like `IOptions<EmailSettings>`. Never put secrets (DB passwords, API keys) in Git. Use User Secrets on your own machine, and environment variables or AWS Secrets Manager / Parameter Store in production.

### Q17. Authentication vs authorization — and how does JWT work?

**One line:** Authentication = checking *who* you are. Authorization = checking *what* you are allowed to do. A JWT is a signed token the client sends with each request to prove who it is.

**Descriptive:** After login, the server creates a JWT with three parts: a header, a payload (claims like user ID, role and expiry time) and a signature. The client sends it with every request as `Authorization: Bearer <token>`. The server checks the signature and expiry — it doesn't need to look up a session, so it scales easily. Use `[Authorize]` or `[Authorize(Roles = "Admin")]` to protect endpoints.

- No token or bad token → **401 Unauthorized**.
- Good token, but no permission → **403 Forbidden**.

Keep tokens short-lived and use refresh tokens. Always use HTTPS. Don't put secret data in the payload — it's only encoded, not encrypted, so anyone can read it.

### Q18. How do you log and monitor a production API?

**One line:** Use `ILogger` with structured logs and the right log levels, send logs to one central place (like AWS CloudWatch), and add health checks and alerts.

**Descriptive:** Inject `ILogger<T>` and log like this: `_logger.LogError(ex, "Payment failed for order {OrderId}", orderId)`. This keeps `OrderId` as a searchable field. Log levels:

- **Debug/Trace** — details for developers.
- **Information** — normal events.
- **Warning** — something odd, but handled.
- **Error** — something failed.
- **Critical** — the system is down.

Tools like Serilog can send logs to files, Seq or CloudWatch. Add a *correlation ID* to each request so you can follow one request through all services. Add a `/health` endpoint for load balancers, and set up dashboards and alerts for errors, slow responses and high CPU. Never log passwords, tokens or card numbers.

## REST APIs (Q19–24)

### Q19. What is a REST API? What makes an API RESTful?

**One line:** A REST API lets clients work with *resources* (like users or orders) using URLs and HTTP methods (GET, POST, PUT, DELETE), usually sending JSON. Each request is independent (stateless).

**Descriptive:** The main rules of REST:

- **Client–server** — the frontend and backend are separate.
- **Stateless** — every request carries everything the server needs (like the JWT token). The server doesn't remember earlier requests. This makes it easy to add more servers.
- **Cacheable** — responses can be cached when it's safe.
- **Uniform interface** — URLs are nouns (`/api/orders/42`), HTTP methods are the actions, and standard status codes show the result.
- **Layered system** — load balancers or caches can sit in the middle without the client knowing.

REST is different from SOAP (XML, strict rules) and GraphQL (one endpoint, the client chooses which fields it wants).

### Q20. Explain the HTTP methods. What is idempotency? PUT vs PATCH?

**One line:** GET reads, POST creates, PUT replaces, PATCH updates part of it, DELETE removes. *Idempotent* means doing the same request many times gives the same result as doing it once.

**Descriptive:** GET, PUT and DELETE are idempotent — deleting order 42 twice still leaves it deleted. POST is not — sending it twice can create two orders. That's why payment APIs use an *idempotency key*. GET is also *safe*: it should never change data. **PUT** sends the whole object and replaces it (missing fields may become empty). **PATCH** sends only the fields you want to change, like `{ "status": "Shipped" }`.

### Q21. Which HTTP status codes should you know?

**One line:** 2xx = success, 3xx = redirect, 4xx = the client made a mistake, 5xx = the server made a mistake.

| Code | Meaning | When to use it |
| --- | --- | --- |
| 200 OK | Success | GET or PUT that returns data |
| 201 Created | New item created | POST; also send a `Location` header |
| 204 No Content | Success, nothing to return | DELETE, or an update with no response body |
| 400 Bad Request | Input is wrong | Validation errors |
| 401 Unauthorized | Not logged in | Token missing or expired |
| 403 Forbidden | Logged in, but not allowed | User doesn't have the right role |
| 404 Not Found | Item doesn't exist | `GET /orders/9999` |
| 409 Conflict | Clashes with current data | Email already used, two users editing at once |
| 422 Unprocessable Entity | Format OK, but breaks a rule | Business rule failed |
| 429 Too Many Requests | Too many calls | Client hit the rate limit |
| 500 Internal Server Error | Bug on the server | Never show the stack trace to users |
| 503 Service Unavailable | Server is down or too busy | During deploys or outages |

### Q22. How would you design endpoints for a resource? How do you handle pagination and versioning?

**One line:** Use plural nouns (`/orders`), nest related items (`/customers/5/orders`), use query parameters for filters and pages, and put a version in the URL (`/v1/`).

**Descriptive:** Example for orders:

- `GET /api/v1/orders?status=pending&page=2&pageSize=20` — list with filter and page
- `GET /api/v1/orders/42` — get one
- `POST /api/v1/orders` — create
- `PATCH /api/v1/orders/42` — update part
- `DELETE /api/v1/orders/42` — delete

Don't put verbs in URLs (`/getOrders`). Never return a huge list in one go — use pages (`page`, `pageSize` and a total count). For very big tables, *cursor* pagination (`?after=1050`) stays fast. Versions (`/v1/`, `/v2/`) let you change the API without breaking old apps that still use it. Document the API with Swagger (OpenAPI).

### Q23. What is CORS and why do you sometimes get CORS errors?

**One line:** CORS is a browser security rule. It blocks a website from calling an API on a different domain unless the API says that website is allowed.

**Descriptive:** If the frontend is on `https://app.example.com` and the API is on `https://api.example.com`, the browser first sends an `OPTIONS` request to ask permission. It then checks the `Access-Control-Allow-Origin` header in the reply. In ASP.NET Core, you set allowed websites with `builder.Services.AddCors(...)` and turn it on with `app.UseCors(...)`. Don't allow every website (`AllowAnyOrigin`) in production. Postman doesn't check CORS — only browsers do. That's why an API can work in Postman but fail in the browser.

### Q24. Walk me through what happens when a client calls `GET /api/orders/42`.

**One line:** The request travels to the server, passes through the middleware, reaches the controller, gets data from the database, and returns as JSON.

**Descriptive:**

1. The client finds the server's address (DNS) and connects over HTTPS, often through a load balancer (like AWS ALB).
2. The web server (Kestrel) receives the request. Middleware runs: error handling, HTTPS, authentication (checks the JWT) and authorization.
3. Routing finds `OrdersController.GetById(int id)` and sets `id = 42`.
4. The controller calls `IOrderService`. The service may check Redis first; if the data isn't there, it asks SQL Server or PostgreSQL through EF Core.
5. The data is turned into a DTO and then into JSON. The server returns 200 OK (or 404 if not found), and the logs record the status and time taken.

## SQL Server, PostgreSQL & database design (Q25–32)

### Q25. Explain the different types of JOINs.

**One line:** INNER JOIN gives only matching rows. LEFT JOIN gives all rows from the left table plus matches. RIGHT JOIN is the opposite. FULL OUTER JOIN gives all rows from both. CROSS JOIN gives every combination.

**Descriptive:** Take tables `Customers` and `Orders`. An INNER JOIN shows only customers who have orders. A LEFT JOIN shows all customers; for customers with no orders, the order columns are NULL. This is useful to find customers who never ordered:

```sql
SELECT c.Id, c.Name
FROM Customers c
LEFT JOIN Orders o ON o.CustomerId = c.Id
WHERE o.Id IS NULL;
```

A SELF JOIN joins a table with itself (for example, employees and their managers). Be careful: joining one row to many rows can repeat data before you add things up.

### Q26. What is normalization? Explain 1NF, 2NF and 3NF.

**One line:** Normalization means splitting data into related tables so the same data isn't stored in many places.

**Descriptive:**

- **1NF** — each column holds one value only. No lists like "017..., 018..." in one phone column.
- **2NF** — 1NF, and every column depends on the *whole* primary key (this matters when the key has more than one column).
- **3NF** — 2NF, and no column depends on another non-key column. Example: don't store `CustomerCity` in `Orders`; it belongs in `Customers`.

Benefits: less repeated data and fewer update mistakes. *Denormalization* is the opposite — you add some repeated data on purpose (like a saved `OrderTotal`) to make reading faster, for example in reports.

### Q27. What are primary keys, foreign keys and other constraints?

**One line:** A primary key uniquely identifies each row. A foreign key links to another table's primary key and makes sure the link is valid.

**Descriptive:** A primary key can't be repeated or NULL. It's usually an auto-increment `INT` or a `GUID`/`UUID`. A foreign key (like `Orders.CustomerId → Customers.Id`) stops you from adding an order for a customer who doesn't exist. You also choose what happens on delete: block it, delete related rows too (cascade), or set them to NULL. Other constraints:

- `UNIQUE` — no duplicates (e.g. email).
- `NOT NULL` — value is required.
- `CHECK` — a rule, e.g. `Price > 0`.
- `DEFAULT` — a default value, e.g. `CreatedAt = GETUTCDATE()`.

Constraints protect the data even if the app code has a bug.

### Q28. What is an index? Clustered vs non-clustered?

**One line:** An index helps the database find rows fast without reading the whole table — like the index at the back of a book. A clustered index decides how the table itself is sorted; a non-clustered index is a separate lookup list.

**Descriptive:** A table can have only one clustered index (in SQL Server, usually on the primary key) but many non-clustered ones. Add indexes on columns you use in `WHERE`, `JOIN` and `ORDER BY`. A *covering index* contains all the columns a query needs (`INCLUDE` in SQL Server), so the database doesn't need extra lookups. Column order matters: an index on `(CustomerId, OrderDate)` helps searches by `CustomerId`, but not by `OrderDate` alone. Downsides: indexes use extra space and make INSERT, UPDATE and DELETE slower — so don't index every column. PostgreSQL also has special indexes like GIN (for JSON and text search) and partial indexes.

### Q29. WHERE vs HAVING — and what's the logical order of a SQL query?

**One line:** WHERE filters rows *before* grouping. HAVING filters groups *after* grouping.

**Descriptive:** The database runs a query in this order: `FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → TOP/LIMIT`. That's why you can't use a SELECT alias inside WHERE. Example — departments with more than 5 employees earning over 50,000:

```sql
SELECT DepartmentId, COUNT(*) AS EmpCount
FROM Employees
WHERE Salary > 50000
GROUP BY DepartmentId
HAVING COUNT(*) > 5;
```

A common follow-up is finding the second-highest salary: `SELECT MAX(Salary) FROM Employees WHERE Salary < (SELECT MAX(Salary) FROM Employees);` — or use `DENSE_RANK() OVER (ORDER BY Salary DESC)` and pick rank 2.

### Q30. What is a transaction? Explain ACID and isolation levels.

**One line:** A transaction is a group of steps that must all succeed together or all fail together. ACID = Atomicity, Consistency, Isolation, Durability.

**Descriptive:** Example: a bank transfer takes money from one account and adds it to another. If adding fails, taking must be undone (`BEGIN TRAN … COMMIT` or `ROLLBACK`).

- **Atomicity** — all or nothing.
- **Consistency** — the data follows all rules before and after.
- **Isolation** — transactions running at the same time don't see each other's half-finished work.
- **Durability** — once saved, data stays even if the server crashes.

Isolation levels balance safety and speed: Read Uncommitted (can see unsaved data), Read Committed (the default in SQL Server and PostgreSQL), Repeatable Read, and Serializable (safest, slowest). SQL Server also has Snapshot. Keep transactions short to avoid locks and deadlocks. In EF Core, `SaveChanges()` already uses a transaction.

### Q31. What differences between SQL Server and PostgreSQL should a developer know?

**One line:** Both are relational databases using SQL, but some syntax, data types and features are different.

| Topic | SQL Server | PostgreSQL |
| --- | --- | --- |
| Get first N rows | `SELECT TOP 10` / `OFFSET … FETCH` | `LIMIT 10 OFFSET 20` |
| Auto-increment | `IDENTITY(1,1)` | `GENERATED AS IDENTITY` or `SERIAL` |
| Unicode text | `NVARCHAR` | `TEXT` / `VARCHAR` (UTF-8 by default) |
| Current time | `GETDATE()`, `SYSUTCDATETIME()` | `NOW()`, `CURRENT_TIMESTAMP` |
| Upper/lower case | Usually ignores case by default | Case matters; use `ILIKE` or `citext` |
| JSON | Stored as `NVARCHAR` + JSON functions | Built-in `JSONB` type with indexes |
| Procedure language | T-SQL | PL/pgSQL |
| Cost | Paid (free Express/Developer editions) | Free and open source |

With EF Core you mostly just switch the provider (`UseSqlServer` or `UseNpgsql`). But raw SQL, upper/lower case and naming rules (PostgreSQL makes unquoted names lowercase) can still cause problems.

### Q32. A query is slow. How do you optimize it?

**One line:** First measure using the execution plan, then fix the real cause — usually a missing index, reading too much data, or a condition that can't use an index.

**Descriptive:**

1. Run the query and measure it (`SET STATISTICS IO, TIME ON` in SQL Server; `EXPLAIN ANALYZE` in PostgreSQL).
2. Read the execution plan. Look for full table *scans* on big tables, and big differences between estimated and actual row counts.
3. Add or fix indexes on the columns used for filtering and joining.
4. Select only the columns you need, not `SELECT *`. Use paging.
5. Write conditions that can use an index. `WHERE YEAR(OrderDate) = 2026` can't use an index, but `WHERE OrderDate >= '2026-01-01' AND OrderDate < '2027-01-01'` can. Avoid `LIKE '%abc'` (a `%` at the start).
6. In the app code, fix N+1 queries and cache data that is read often but rarely changes (Redis).

**Quick one-liners you may also get:**

- **DELETE vs TRUNCATE vs DROP:** DELETE removes chosen rows (can use WHERE). TRUNCATE quickly removes all rows and resets the ID counter. DROP removes the whole table.
- **UNION vs UNION ALL:** UNION removes duplicates (slower). UNION ALL keeps them.
- **Stored procedure vs function:** a procedure can change data and return several results. A function returns a value or table and can be used inside a query.
- **View:** a saved SELECT query that you can use like a table.

## Entity Framework Core (Q33–35)

### Q33. What is Entity Framework Core? Explain DbContext, DbSet and migrations.

**One line:** EF Core is an ORM from Microsoft. It connects C# classes to database tables, so you can use LINQ instead of writing SQL by hand.

**Descriptive:**

- **`DbContext`** — your connection to the database. It tracks changes and saves them with `SaveChanges()`.
- **`DbSet<Order>`** — represents one table (here, the Orders table).
- **Code-First** — you write the C# classes first, and EF creates the tables from them.
- **Migrations** — files that describe database changes step by step. Create one with `dotnet ef migrations add AddOrderStatus`, then apply it with `dotnet ef database update`. Migrations are saved in Git, so every developer and server gets the same database structure.
- **Database-First** — the opposite: EF creates C# classes from an existing database.

The same code can work with SQL Server (`UseSqlServer`) or PostgreSQL (`UseNpgsql`) by changing the provider.

### Q34. Eager, lazy and explicit loading — and what is the N+1 problem?

**One line:** Eager loading gets related data at the start with `Include()`. Lazy loading gets it automatically the first time you use it. Explicit loading gets it when you ask with `Entry().Load()`. The N+1 problem is 1 query for a list plus 1 extra query for every item.

**Descriptive:** Example: you load 100 orders, then loop and read `order.Customer.Name`. With lazy loading, that's 1 + 100 = 101 queries. This is the N+1 problem, and it makes the app very slow. Fixes: use `_db.Orders.Include(o => o.Customer)` (one query with a JOIN), or better, select only the fields you need:

```csharp
var result = await _db.Orders
    .Where(o => o.Status == OrderStatus.Pending)
    .Select(o => new OrderDto(o.Id, o.Total, o.Customer.Name))
    .ToListAsync();
```

For read-only queries, add `AsNoTracking()`. EF then skips change tracking, so it's faster and uses less memory. If many `Include`s create a huge result, use `AsSplitQuery()` to split it into smaller queries.

### Q35. How does EF Core change tracking work, and how do you handle concurrent updates?

**One line:** EF remembers the state of each object (Added, Modified, Deleted, Unchanged) and creates the right SQL when you call `SaveChanges()`. To handle two people editing the same row, use a concurrency token like `RowVersion`.

**Descriptive:** When you load an object and change a property, EF marks it as Modified and updates only the changed columns. Problem: if two users edit the same record, the second save silently overwrites the first. The fix is *optimistic concurrency*: add `[Timestamp] public byte[] RowVersion { get; set; }` (SQL Server), or use `xmin` in PostgreSQL. EF adds this value to the UPDATE's WHERE. If someone else already changed the row, no rows are updated and EF throws `DbUpdateConcurrencyException`. You catch it and return 409 Conflict. For complex reports or bulk updates, raw SQL (`FromSqlInterpolated`, which is safe from SQL injection) or Dapper can be faster. Never build SQL by joining strings.

## Git, Redis, AWS & Jenkins (Q36–40)

### Q36. Describe your Git workflow. Merge vs rebase?

**One line:** Make a new branch from `main`, make small commits, push, open a pull request (PR), get it reviewed, then merge it.

**Descriptive:** A normal flow:

1. `git checkout -b feature/order-export` — create a branch.
2. Write code, then `git add` and `git commit -m "Add CSV export for orders"`.
3. `git pull --rebase origin main` — get the latest changes from main.
4. `git push`, then open a PR. After review and passing tests, merge it.

Write clear commit messages and keep commits small. **Merge** joins two branches and adds a merge commit, keeping the full history. **Rebase** moves your commits on top of the latest `main`, so the history looks like one straight line. But rebase changes commit history, so never rebase a branch other people are using. Many teams *squash merge*, which turns all PR commits into one commit on `main`.

### Q37. How do you resolve a merge conflict? What's the difference between `reset`, `revert` and `stash`?

**One line:** A conflict happens when two branches change the same lines. You edit the file to keep the correct code, then `git add` and commit. `revert` safely undoes a commit with a new commit, `reset` moves your branch back, and `stash` saves unfinished work for later.

**Descriptive:** Git marks conflicts with `<<<<<<<`, `=======` and `>>>>>>>`. Understand both changes (ask the other developer if unsure), keep the right code, remove the markers, build and run the tests, then `git add` and `git commit` (or `git rebase --continue`).

- `git revert <hash>` — undo a commit by adding a new one. Safe on shared branches.
- `git reset --soft` — undo commits but keep your changes. `--hard` throws the changes away — be careful.
- `git stash` — save unfinished work so you can switch branches (e.g. for an urgent fix). Get it back with `git stash pop`.

Other useful commands: `git log --oneline`, `git diff`, `git cherry-pick`, `git blame`.

### Q38. What is Redis and how would you use it in a .NET API?

**One line:** Redis is a very fast in-memory key-value store. It's mostly used as a cache, and also for sessions, rate limiting, messaging and locks.

**Descriptive:** Reading from memory is much faster than a database query. The common pattern is **cache-aside**:

1. Check Redis for a key like `product:42`.
2. If found, return it.
3. If not, read it from the database, save it in Redis with an expiry time (TTL, e.g. 10 minutes), then return it.

When the product changes, delete or update the Redis key so users don't see old data (*cache invalidation*). In .NET, use `StackExchange.Redis` (register `ConnectionMultiplexer` as a Singleton) or the `IDistributedCache` interface. Redis supports strings, hashes, lists, sets and sorted sets (good for leaderboards). Cache data that is read often and changes rarely. Redis is a helper — the database is still the main source of truth. On AWS, Redis is available as ElastiCache.

### Q39. Which AWS services might a .NET SaaS backend use?

**One line:** Usually EC2, ECS/Fargate or Lambda to run the code; RDS for SQL Server/PostgreSQL; ElastiCache for Redis; S3 for files; and CloudWatch for logs and monitoring.

**Descriptive:**

- **Running code:** EC2 (virtual machines), Elastic Beanstalk (easy .NET hosting), ECS/Fargate (Docker containers), Lambda (small functions without managing servers).
- **Data:** RDS (managed SQL Server/PostgreSQL with backups), ElastiCache (Redis), S3 (file storage).
- **Network:** VPC, Application Load Balancer, Route 53 (DNS), CloudFront (CDN).
- **Security:** IAM roles with minimum permissions; Secrets Manager / Parameter Store for passwords and connection strings.
- **Messaging:** SQS (queues for background jobs like sending emails), SNS (notifications).
- **Monitoring:** CloudWatch logs, metrics and alarms.

As a fresher, it's fine to say you know the basics, maybe used the free tier, and can explain what each service does.

### Q40. What is CI/CD, and how does a Jenkins pipeline work?

**One line:** CI (Continuous Integration) automatically builds and tests every code change. CD (Continuous Delivery/Deployment) automatically releases builds that pass. Jenkins runs these steps from a `Jenkinsfile`.

**Descriptive:** When a developer pushes code or opens a PR, Jenkins starts automatically. A typical .NET pipeline:

1. **Checkout** — get the code from Git.
2. **Restore** — `dotnet restore` (download packages).
3. **Build** — `dotnet build -c Release`.
4. **Test** — `dotnet test`, and save the results.
5. **Package** — `dotnet publish` or build a Docker image.
6. **Deploy** — to a staging server (e.g. on AWS), then after approval, to production.

If any step fails, the pipeline stops and the team is notified, so broken code doesn't reach users. Benefits: fast feedback, the same deployment every time, and fewer "it works on my machine" problems. Similar tools: GitHub Actions, GitLab CI, Azure DevOps.

## Unit testing, code reviews & DSA (Q41–43)

### Q41. How do you write unit tests in .NET? What is mocking?

**One line:** Use xUnit (or NUnit/MSTest) to test one small piece of logic at a time, using the Arrange–Act–Assert pattern. Use Moq or NSubstitute to create fake versions (mocks) of other parts.

**Descriptive:** A unit test checks one method without using a real database, network or files. That's why it's fast and reliable. Because services depend on interfaces (thanks to DI), you can give them a fake object:

```csharp
[Fact]
public async Task GetOrder_ReturnsNull_WhenOrderDoesNotExist()
{
    // Arrange - set things up
    var repo = new Mock<IOrderRepository>();
    repo.Setup(r => r.GetByIdAsync(99)).ReturnsAsync((Order?)null);
    var service = new OrderService(repo.Object);

    // Act - run the method
    var result = await service.GetOrderAsync(99);

    // Assert - check the result
    Assert.Null(result);
}
```

Test the normal case, edge cases (empty list, zero, null) and error cases. Use `[Theory]` with `[InlineData]` to test many inputs. Name tests by what they check (`Method_ExpectedResult_WhenCondition`). *Integration tests* go further — they use a real test database or call the whole API with `WebApplicationFactory`. Focus on testing important business logic; you don't need 100% coverage.

### Q42. What do you look for in a code review — and how do you handle feedback on your own code?

**One line:** I check if the code is correct, easy to read, tested, secure and fast enough. When I get feedback, I see it as a chance to learn, ask questions, and don't take it personally.

**Descriptive:** When reviewing, I ask:

- Does it do what was asked, and handle edge cases?
- Is it easy to read — clear names, small methods, no copy-paste code?
- Are there tests?
- Any security problems — SQL injection, missing `[Authorize]`, passwords in the code?
- Any performance problems — N+1 queries, no paging, blocking async code?
- Does it follow the team's style?

Comments should be specific and kind, and about the code, not the person. Suggest instead of ordering: "Could we move this into a separate method?" When it's my code: keep PRs small with a clear description, review it myself first, reply to every comment, and ask "why" if I don't understand. For a junior developer, code reviews are one of the fastest ways to learn.

### Q43. What is Big-O notation? Which data structure would you choose for fast lookups?

**One line:** Big-O shows how the time or memory needed grows as the input grows. For fast lookups by key, use a hash table — `Dictionary<TKey,TValue>` or `HashSet<T>` — which takes O(1) time on average.

**Descriptive:** Common Big-O values, from fastest to slowest:

- **O(1)** — constant, e.g. a dictionary lookup.
- **O(log n)** — e.g. binary search.
- **O(n)** — one loop through the data.
- **O(n log n)** — good sorting algorithms.
- **O(n²)** — a loop inside a loop.

Know the C# collections: `List<T>` (fast by index, slow to search), `Dictionary` (fast by key), `HashSet` (unique items, fast `Contains`), `Queue`/`Stack` (first-in-first-out / last-in-first-out). Classic example — *Two Sum* (find two numbers that add up to a target): two loops is O(n²). One loop with a dictionary that stores each number and its index, checking if `target - current` is already there, is O(n). Also be ready to talk about arrays vs linked lists, recursion, binary search, and BFS vs DFS.

## Scenario-based questions (Q44–50)

For every scenario, follow the same steps: **understand the problem → investigate → fix → test → stop it from happening again → keep people informed.**

### Q44. Right after a deployment, users report that an endpoint returns 500 errors. What do you do?

**One line:** Check how bad it is, look at the logs to find the error, roll back if it's serious, then reproduce it, fix it, test it, and add a test so it doesn't happen again.

**Descriptive:**

1. Tell the team and my senior first — don't quietly try to fix a production problem alone.
2. Check how many users are affected.
3. Look at the logs (e.g. CloudWatch) for the error message and stack trace.
4. Check what changed in this deployment: the code, settings, a new database migration, environment variables.
5. If many users are affected, the fastest safe fix is usually to **roll back** to the previous version, then investigate calmly.
6. Common causes: a missing setting or secret in production, a migration that wasn't applied, a null value in real data that test data didn't have, or a missing DI registration.
7. Reproduce it on my machine or staging, fix it, add a unit test for that case, get it reviewed, and deploy again.

Afterwards, join a short review meeting (without blaming anyone): what happened, why tests didn't catch it, and what we'll change.

### Q45. An API that used to respond in 200 ms now takes 5 seconds. How do you investigate?

**One line:** Find which part is slow — usually the database — then check the query plan, data growth, indexes and N+1 queries.

**Descriptive:** First check monitoring: is every request slow or only some? Did it start after a deployment, or slowly as data grew? Add timing logs or use a profiler to see where the time goes — the database, an external API call, or the code. Turn on EF Core SQL logging to see the actual queries. Maybe a new `Include` made a huge join, a loop is causing N+1 queries, or a `.ToList()` before `.Where()` loads the whole table. Run the slow query and read its execution plan to find scans and missing indexes. Also check database locks and the server's CPU and memory. Possible fixes: add an index, select fewer columns with `Select`, add paging, use `AsNoTracking`, cache in Redis, or move slow work to a background job. Measure before and after to prove the fix worked.

### Q46. Customers sometimes get charged twice or get duplicate orders when they double-click "Place order". How do you fix it?

**One line:** Make the server safe against repeated requests using an idempotency key and a unique constraint — and also disable the button in the frontend.

**Descriptive:** Disabling the button helps, but it isn't enough — network retries or bad clients can still send the request twice. On the backend:

- The client sends a unique `Idempotency-Key` (a GUID) with the request.
- The server saves this key with a unique constraint. If the same key comes again, the server returns the first result instead of creating a new order.
- The database unique constraint is the final safety net. Without it, two requests arriving at the same moment could both check "does it exist?", both see "no", and both create an order.
- Save the order and payment record inside one transaction.

I'd also work with the frontend and QA teams so they can reproduce the double-click and confirm the fix.

### Q47. QA reports a bug that you can't reproduce on your machine. What do you do?

**One line:** Get the exact steps, data and environment from QA, compare their setup with mine, and check the logs — never just say "it works for me".

**Descriptive:** Ask QA for: the exact steps, the user account and test data, which environment (staging or local), the browser, the time it happened, the request and response from the browser's network tab, and screenshots. Differences usually come from:

- **Data** — a null field, special characters, time zones.
- **Settings** — different configuration.
- **Database** — SQL Server on my machine vs PostgreSQL on staging (e.g. upper/lower case matters in PostgreSQL).
- **Cache** — old data in Redis.
- **Timing** — two requests at the same time.

Check the logs for that time. Try to reproduce on the same environment with the same user. If it happens only sometimes, add more logging. Once found, write a test that fails because of the bug, fix it, and ask QA to verify. QA are teammates — they help make the product better.

### Q48. You're given a feature with unclear requirements, or you've been stuck on a task for hours. What do you do?

**One line:** Ask the product owner early to clear things up, write down what I understood, and ask a senior for help after a fixed amount of time instead of staying stuck quietly.

**Descriptive:** For unclear requirements, I list my questions and assumptions — edge cases, validation rules, who can use the feature, what the error messages should be — and confirm them with the product manager *before* coding. Building the wrong thing is the most expensive mistake. I also agree on the API request/response format with the frontend developer early, so we can both work at the same time. When I'm stuck, I set a time limit (e.g. 30–60 minutes). I search the docs and the codebase, then ask a senior with a clear summary: what I'm trying to do, what I tried, and the error I got. If a deadline is at risk, I tell my lead early — not on the last day.

### Q49. You accidentally pushed a commit containing an AWS secret key, or broke the `main` build. What do you do?

**One line:** Admit it right away, tell the team, cancel and replace the secret immediately, fix or undo the broken build — and then make sure it can't happen again.

**Descriptive:** For a leaked secret: just deleting the commit is not enough, because it's still in the Git history and someone may have seen it.

1. Tell my lead immediately.
2. **Disable and replace (rotate)** the key in AWS IAM, and check if it was misused.
3. Remove it from the code and move it to Secrets Manager or environment variables.
4. The team may also clean it out of the Git history.
5. Add `.gitignore` rules and a secret-scanning tool.

For a broken build: tell the team, then quickly push a fix or `git revert` the commit so others aren't blocked. Then ask why CI didn't catch it — maybe add a required check before merging. Interviewers want to hear honesty and ownership, not hiding mistakes.

### Q50. Design a "Forgot password" feature for our SaaS, end to end.

**One line:** Build two endpoints — one sends an email with a one-time, time-limited link, and one checks that link and sets the new password — built securely and planned with the frontend, QA and product teams.

**Descriptive:**

1. **Ask product:** How long is the link valid (e.g. 15–30 minutes)? What are the password rules? Should the user be logged out on other devices? What should the email say?
2. **Database:** a `PasswordResetTokens` table with `UserId`, `TokenHash`, `ExpiresAt`, `UsedAt`. Save only a *hash* of the token, never the real token.
3. **Endpoint 1 —** `POST /api/auth/forgot-password` with `{ email }`. Always return the same 200 response, whether the email exists or not, so attackers can't find out which emails are registered. If the user exists, create a secure random token, save its hash, and send the email in the background (e.g. through SQS and a background worker).
4. **Endpoint 2 —** `POST /api/auth/reset-password` with `{ token, newPassword }`. Hash the token, find a matching row that isn't expired or used, hash the new password with a strong method (ASP.NET Core Identity's hasher, bcrypt or Argon2), mark the token as used, and log the user out of other sessions.
5. **Security:** limit how often someone can request resets (Redis counters work well), use HTTPS only, and log the event — but never log the token.
6. **Testing:** unit tests for expired, used, invalid and valid tokens. QA tests the full email flow on staging.
7. **Teamwork:** agree on the API format and error messages with the frontend, document it in Swagger, and release it through code review and the CI/CD pipeline.

**Also prepare for these common questions:** a 60-second self-introduction; a walkthrough of one .NET project (the problem, the tech stack, your role, one challenge you solved, what you'd improve — with the GitHub link); why this company and SaaS interest you; and one or two questions to ask them (e.g. "How do new engineers get started here?" or "How do code reviews work on the team?").

# Part 2 — 100 more fundamentals questions (Q51–150)

Part 2 drills the fundamentals the circular lists under "What we're looking for": OOP, data structures and algorithms, C#, basic SQL and relational design, and REST — the areas an entry-level AI interview probes most. Answers here are shorter: say the one-liner, then add the detail if asked.

## OOP deep dive (Q51–65)

### Q51. What is the difference between a class and an object?

**One line:** A class is a blueprint (a plan). An object is a real thing made from that blueprint using `new`.

**Descriptive:** `class Car { public string Model; public void Drive() {...} }` is the plan. `var myCar = new Car();` creates an actual car object in memory with its own data. You can create many objects from one class, and each one can have different values.

### Q52. What is a constructor? What kinds exist in C#?

**One line:** A constructor is a special method that runs automatically when you create an object. It has the same name as the class and no return type. It sets up the object's starting values.

**Descriptive:** Kinds of constructors:

- **Default** — takes no parameters. C# adds one for you only if you don't write any.
- **Parameterized** — takes values, e.g. `new Car("Toyota")`.
- **Static** — runs only once, before the class is used the first time. Used to set static fields.
- **Private** — stops other code from creating objects. Used in the Singleton pattern.

One constructor can call another: `: this(...)` calls another constructor in the same class, and `: base(...)` calls the parent class's constructor. The parent's constructor always runs first.

### Q53. Fields vs properties — why use properties?

**One line:** A field is a plain variable inside a class. A property wraps a field with `get` and `set`, so you control who can read or change it and can add checks.

**Descriptive:** `public int Age { get; private set; }` — anyone can read it, but only the class can change it. A property can also check values: `set { if (value < 0) throw new ArgumentException(); _age = value; }`. An `init` property can only be set when the object is created, which helps make objects that can't be changed later. Public fields break encapsulation, so avoid them.

### Q54. Does C# support multiple inheritance? Why not?

**One line:** Not for classes — a class can inherit only one class. But it can implement many interfaces.

**Descriptive:** Multiple class inheritance causes the *diamond problem*. Say B and C both inherit from A and both change a method `Print()`. If D inherits from both B and C, which `Print()` should D use? C# avoids this confusion. Interfaces give you many "contracts" without this problem. C# supports single inheritance (B from A), multilevel (C from B from A) and hierarchical (B and C both from A).

### Q55. Compile-time vs runtime polymorphism — how does runtime dispatch work?

**One line:** Compile-time polymorphism is method overloading — the compiler picks the method. Runtime polymorphism is method overriding — the program picks the method while running, based on the real object.

**Descriptive:** `Animal a = new Dog(); a.Speak();` calls `Dog.Speak()` if `Speak` is `virtual` in `Animal` and overridden in `Dog`. Behind the scenes, .NET keeps a table of methods for each class (called a *vtable*) to find the right version. Without `virtual`, C# uses the variable's type (`Animal`) instead. Operator overloading (like making `+` work for your class) is also compile-time polymorphism.

### Q56. What is an abstract method? Can an abstract class have a constructor?

**One line:** An abstract method has no code body — child classes must write it. Yes, an abstract class can have a constructor; child classes call it with `base()`.

**Descriptive:** You can't create an object of an abstract class directly — `new Shape()` gives an error. But when you create a `Circle`, the `Shape` constructor still runs to set up shared fields. An abstract class can have both abstract methods and normal methods with code. If a class has even one abstract method, the class itself must be `abstract`.

### Q57. What is a `sealed` class and why would you use it?

**One line:** A sealed class can't be inherited. A sealed method can't be overridden again.

**Descriptive:** Use it when a class isn't meant to be extended — to protect important logic (for example, security code) and to make your intention clear. It can also make code a little faster. The `string` class in .NET is sealed.

### Q58. Static vs instance members — what is a static class?

**One line:** Static members belong to the class itself and are shared by everyone. Instance members belong to each object. A static class can't be used to create objects and has only static members.

**Descriptive:** `Math.Max()` is static — you don't need an object to use it. Static classes are good for helper methods and extension methods. Static methods can't use instance fields directly. Be careful with static variables that change in a web app: they're shared by all requests and threads, which can cause bugs, and they make unit testing harder.

### Q59. Composition vs inheritance — why "favor composition"?

**One line:** Inheritance means "is a" (a Dog *is an* Animal). Composition means "has a" (a Car *has an* Engine) — a class holds other objects as fields. Composition is more flexible.

**Descriptive:** Deep inheritance chains break easily — a change in the parent can break every child class. With composition, a `Car` has an `IEngine`, so you can easily swap a `PetrolEngine` for an `ElectricEngine`, or use a fake engine in tests. Use inheritance only when the child really is a special type of the parent.

### Q60. Explain association, aggregation and composition.

**One line:** All three are "has/uses" relationships. Association is a loose link. Aggregation is "has a", but the part can live on its own. Composition is "has a", and the part dies with the whole.

**Descriptive:**

- **Association:** a `Teacher` teaches a `Student`.
- **Aggregation:** a `Department` has `Professors`, but professors still exist if the department closes.
- **Composition:** a `House` has `Rooms`; if the house is destroyed, the rooms are too.

In a database, this is like choosing `ON DELETE CASCADE` (composition) or `SET NULL` (aggregation).

### Q61. What do the `this` and `base` keywords do?

**One line:** `this` means the current object. `base` means the parent class.

**Descriptive:** `this.name = name;` tells apart the field from the parameter with the same name. `this(...)` calls another constructor, and `this` is also used to write extension methods. `base.Save()` calls the parent's version of a method from inside an override — for example, to add logging and then run the original code. `base(...)` calls the parent's constructor.

### Q62. What is explicit interface implementation?

**One line:** Writing an interface method with the interface name in front, like `void ILogger.Log()`. Then it can only be called through the interface.

**Descriptive:** It's useful when a class implements two interfaces that have a method with the same name but different meanings. It also hides methods you don't want in the class's normal public list. To call it, you must cast: `((ILogger)obj).Log()`.

### Q63. What are coupling and cohesion?

**One line:** Aim for low coupling (classes don't depend too much on each other) and high cohesion (everything in a class works toward one clear purpose).

**Descriptive:** A `UserService` that does validation, SQL, sending emails and making PDFs has low cohesion — it does too many things. A controller that directly creates `new SqlConnection(...)` is tightly coupled to SQL Server. Interfaces, dependency injection and the Single Responsibility Principle help keep coupling low and cohesion high, so code is easier to test and change.

### Q64. Which design patterns do you know? Give examples.

**One line:** Common ones: Singleton, Factory, Repository, Strategy, Observer, Decorator and Dependency Injection.

**Descriptive:**

- **Singleton** — only one object in the whole app. In ASP.NET Core, use `AddSingleton` instead of writing it yourself.
- **Factory** — one place decides which class to create, e.g. `PaymentProcessorFactory.Create("card")`.
- **Repository** — hides database code behind an interface (`IOrderRepository`), which makes testing easier.
- **Strategy** — different ways of doing the same job behind one interface, e.g. different discount or shipping calculators.
- **Observer** — when something happens, everyone who subscribed is told. C# `event`s work like this.
- **Decorator** — wraps an object to add extra behavior, e.g. a caching repository around the real one.

### Q65. What do DRY, KISS and YAGNI mean?

**One line:** DRY — Don't Repeat Yourself. KISS — Keep It Simple. YAGNI — You Aren't Gonna Need It (don't build things before you need them).

**Descriptive:** DRY: put repeated code in one method, so a bug is fixed in one place. KISS: a simple, readable solution is usually better than a clever one. YAGNI: don't add extra features or settings "just in case". These rules balance each other — too much DRY can create confusing shared code.

## C# language fundamentals (Q66–85)

### Q66. `==` vs `.Equals()` — and why override `GetHashCode()` with `Equals()`?

**One line:** For classes, `==` checks by default if two variables point to the *same object*. `Equals()` can be changed to compare the *values* inside. `string` compares the text for both.

**Descriptive:** Two `Customer` objects with the same `Id` are not `==` unless you change how equality works. If you override `Equals()`, you must also override `GetHashCode()`, so equal objects give the same hash code. If you don't, `Dictionary` and `HashSet` won't find your objects correctly. `record` types do value comparison automatically.

### Q67. `const` vs `readonly` vs `static readonly`?

**One line:** `const` is fixed when you compile and never changes. `readonly` is set once — where it's declared or in the constructor. `static readonly` is one shared value, set once when the program runs.

**Descriptive:** `const double Pi = 3.14159;` must be a simple value (number, string) known before the program runs. `readonly` fields can get values at runtime, like `DateTime.Now` or an injected service, and can be different for each object. Use `static readonly` for one shared object, like an `HttpClient` or a regex. Note: if you change a public `const` in a library, other projects using it must be rebuilt, because the old value was copied into them.

### Q68. `ref` vs `out` vs `in` parameters?

**One line:** All three pass a variable by reference. `ref` needs a starting value and can be changed. `out` doesn't need a starting value but the method must set it. `in` can't be changed (read-only).

**Descriptive:** `out` is used in the Try-pattern: `if (int.TryParse(input, out int n))`. `ref` lets a method change the caller's variable (for example, swapping two values). `in` passes big structs without copying them, but stops changes. Often, returning a tuple like `(bool ok, int value)` is cleaner than many `out` parameters.

### Q69. How do nullable types work? Explain `?`, `??`, `?.` and `!`.

**One line:** `int?` lets a number be null. `??` gives a backup value if something is null. `?.` safely uses an object that might be null. `!` tells the compiler "I promise this isn't null".

**Descriptive:** Examples: `int? age = null; int a = age ?? 0;` gives 0 if age is null. `var city = user?.Address?.City;` returns null instead of crashing with a `NullReferenceException`. `??=` sets a value only if it's currently null. With *nullable reference types* turned on (the default in new projects), `string?` means "this may be null", and the compiler warns you if you might use a null value. This helps prevent the most common runtime error.

### Q70. `var` vs `dynamic` vs `object`?

**One line:** `var` lets the compiler figure out the type, but it's still a fixed type. `object` can hold anything but needs casting back. `dynamic` skips type checks until the program runs.

**Descriptive:** `var list = new List<int>();` is still a `List<int>`, with full IntelliSense. `object o = 5;` boxes the number, and you must cast it back to use it. `dynamic d = GetJson(); d.Anything();` compiles even if `Anything` doesn't exist — it fails only when the program runs. Use `dynamic` rarely.

### Q71. When would you use a `struct` instead of a `class`?

**One line:** For small, simple values that don't change — like a point, coordinates or a money amount — where copying is fine and you want to avoid extra memory use.

**Descriptive:** Structs are value types: they're copied when assigned, can't inherit from other classes or structs (but can implement interfaces), and can't be null unless you use `Nullable<T>`. Big or changeable structs cause bugs (you change a copy by mistake) and are slow to copy. Most models and entities should be classes.

### Q72. What is a `record` in C#?

**One line:** A record is a type made for holding data. Two records with the same values are equal, it prints nicely with `ToString()`, and you can make changed copies with `with`.

**Descriptive:** `public record OrderDto(int Id, decimal Total);` creates a class whose values are set when created. `var updated = dto with { Total = 99 };` makes a new copy with one value changed, leaving the original as it was. Records are great for DTOs and API request/response models. `record struct` is the value-type version.

### Q73. What are delegates? Explain `Func`, `Action` and `Predicate`.

**One line:** A delegate is a variable that holds a method. `Func<T, TResult>` returns a value, `Action<T>` returns nothing, and `Predicate<T>` returns true or false.

**Descriptive:** Delegates let you pass a method as a parameter. LINQ uses them: in `list.Where(x => x > 5)`, the lambda `x => x > 5` is a `Func<int, bool>`. Another example: `Action<string> log = msg => Console.WriteLine(msg);`. A delegate can hold several methods at once, and that's how events work.

### Q74. What is an event in C#, and how is it different from a delegate?

**One line:** An event is a protected delegate. Outside code can only subscribe (`+=`) or unsubscribe (`-=`); only the class that owns it can trigger it.

**Descriptive:** It follows the Observer pattern. Example: `public event EventHandler<OrderPlacedEventArgs>? OrderPlaced;` — an email service can subscribe and get notified when an order is placed. With a plain public delegate, outside code could replace it or call it directly; `event` blocks that. If you forget to unsubscribe from a long-living object, it can cause a memory leak.

### Q75. What are extension methods?

**One line:** Static methods that you can call as if they were part of an existing type. You write them in a static class, with `this` before the first parameter.

**Descriptive:** `public static bool IsValidEmail(this string s) => ...;` lets you write `email.IsValidEmail()`. LINQ methods like `Where` and `Select` are extension methods on `IEnumerable<T>`. ASP.NET Core's `services.AddScoped()` registrations also use them. Extension methods can't access private members of the type.

### Q76. What are generics, and what are generic constraints?

**One line:** Generics let you write one piece of code that works with many types safely, using a type placeholder like `T` (e.g. `List<T>`). Constraints (`where T : ...`) limit which types are allowed.

**Descriptive:** Without generics, you'd use `object`, lose type safety and pay boxing costs. Example: `public class Repository<T> where T : class, IEntity, new()` means `T` must be a class, must implement `IEntity`, and must have an empty constructor. Other constraints include `struct`, `notnull`, a base class and `unmanaged`.

### Q77. What is `IDisposable` and the `using` statement? What is a finalizer?

**One line:** `IDisposable.Dispose()` frees resources like database connections and files right away. `using` calls `Dispose()` for you automatically, even if there's an error. A finalizer is a backup cleanup method that the garbage collector calls later.

**Descriptive:** `using var conn = new SqlConnection(cs);` closes the connection when the code block ends. Without it, connections stay open until the garbage collector runs, and the app can run out of connections when busy. Finalizers (`~MyClass()`) run at unknown times and slow things down, so they're only a safety net. In ASP.NET Core, the DI system automatically disposes scoped services (like `DbContext`) at the end of each request.

### Q78. What does `yield return` do?

**One line:** It returns items one by one, only when they're needed, instead of building the whole list in memory first.

**Descriptive:** A method that returns `IEnumerable<int>` and uses `yield return` pauses after giving each item, and continues when the next item is requested. It's useful for reading large files line by line. Like LINQ, nothing runs until you loop over it — and looping twice runs the method twice.

### Q79. Array vs `List<T>` vs `ArrayList`?

**One line:** An array has a fixed size. `List<T>` can grow and is type-safe. `ArrayList` is an old type that stores everything as `object` — avoid it.

**Descriptive:** `int[] a = new int[10];` can't grow. `List<int>` doubles its internal array when it gets full, so adding items is fast on average. `ArrayList` needs casting and boxes value types, so it's slower and less safe. Use arrays when the size is fixed and speed matters; use `List<T>` almost everywhere else.

### Q80. `is` vs `as` vs a direct cast — and what is pattern matching?

**One line:** A direct cast `(Dog)x` crashes if it fails. `as` returns null if it fails. `is` checks the type, and with pattern matching it can check and convert in one step.

**Descriptive:** `if (animal is Dog d) d.Bark();` is the modern, safe way. `switch` expressions can use patterns too: `var fee = order switch { { Total: > 1000 } => 0m, { IsExpress: true } => 10m, _ => 5m };` (free delivery over 1000, 10 for express, otherwise 5). `as` only works with reference types and nullable types.

### Q81. What are enums and when do you use them?

**One line:** An enum is a set of named constant numbers, like `enum OrderStatus { Pending, Paid, Shipped, Cancelled }`. Use it instead of "magic" numbers or strings.

**Descriptive:** Enums make code easier to read and safer. By default they're `int`s starting at 0. If you save enums in a database, give them fixed numbers, so changing the order doesn't break old data. `[Flags]` enums let you combine values: `Permissions.Read | Permissions.Write`. EF Core saves enums as numbers by default, or as text if you configure it.

### Q82. What is a race condition? How do you make code thread-safe?

**One line:** A race condition happens when two threads use the same data at the same time and the result depends on who goes first. Fix it with locks, atomic operations, thread-safe collections, or by not sharing changeable data.

**Descriptive:** `count++` isn't one single step. Two threads can read the same value, and both write back value + 1 — so one increase is lost. Fixes: `lock (_sync) { count++; }`, `Interlocked.Increment(ref count)`, or `ConcurrentDictionary`. A *deadlock* happens when two threads each wait for a lock the other one holds — avoid it by always locking in the same order and keeping locks short. In web apps with several servers, normal C# locks only work inside one server, so use database constraints or Redis locks instead.

### Q83. Task vs Thread — and what do `Task.WhenAll` and `Task.WhenAny` do?

**One line:** A `Thread` is a real operating-system thread you manage yourself. A `Task` is a lighter piece of work that .NET schedules for you, and it works with `async/await`.

**Descriptive:** Prefer tasks — they're cheaper and handle results, errors and cancellation well. `await Task.WhenAll(getUser, getOrders)` runs independent calls at the same time and waits for all of them — faster than running them one after another. `Task.WhenAny` returns when the first one finishes (useful for timeouts). Note: one `DbContext` can't run two queries at the same time, so don't use `WhenAll` with the same `DbContext`.

### Q84. What are `params`, optional and named parameters?

**One line:** `params` lets a method accept any number of arguments. Optional parameters have default values. Named arguments let you pass values by parameter name, in any order.

**Descriptive:** `int Sum(params int[] nums)` can be called as `Sum(1, 2, 3)`. In `void Send(string to, bool urgent = false)`, `urgent` is optional. `Send(to: "a@b.com", urgent: true)` is easier to read, especially for true/false values.

### Q85. If you pass an object to a method and modify it there, does the caller see the change? What if you reassign it?

**One line:** If you change the object's properties — yes, the caller sees it. If you point the parameter to a new object — no, unless you used `ref`.

**Descriptive:** C# gives the method a *copy of the reference* — like a copy of the address to the same house. `p.Name = "New"` changes the house, so the caller sees it. `p = new Person()` only changes the method's copy of the address; the caller still points to the old object. With `ref Person p`, the caller's variable would change too. For value types (like `int` or structs), the whole value is copied, so changes inside the method aren't seen outside without `ref`.

## Data structures (Q86–97)

### Q86. Compare the time complexity of common data structures.

**One line:** Choose a data structure based on what you do most: get by position (array), find by key (hash table), keep things sorted (balanced tree), or first-in-first-out / last-in-first-out (queue / stack).

| Structure (C# type) | Get by index | Search | Add | Remove |
| --- | --- | --- | --- | --- |
| Array (`T[]`) | O(1) | O(n) | O(n) | O(n) |
| Dynamic array (`List<T>`) | O(1) | O(n) | O(1) at the end on average, O(n) in the middle | O(n) |
| Linked list (`LinkedList<T>`) | O(n) | O(n) | O(1) at a known node | O(1) at a known node |
| Stack / Queue | — | O(n) | O(1) push / enqueue | O(1) pop / dequeue |
| Hash table (`Dictionary`, `HashSet`) | — | O(1) on average, O(n) worst | O(1) on average | O(1) on average |
| Balanced tree (`SortedDictionary`, `SortedSet`) | — | O(log n) | O(log n) | O(log n) |
| Heap (`PriorityQueue`) | — | O(n) | O(log n) | O(log n) to remove the smallest |

### Q87. Array vs linked list?

**One line:** An array keeps items side by side in memory, so getting item number 5 is instant, but inserting in the middle is slow. A linked list connects items with pointers, so inserting is fast, but finding item number 5 means walking through the list.

**Descriptive:** Arrays are fast to loop through, so in real life `List<T>` is the best choice most of the time. Linked lists are useful when you often add or remove items in the middle while walking through the list (for example, an LRU cache). Each linked-list item also needs extra memory for its pointers.

### Q88. Stack vs queue — give real examples.

**One line:** A stack is Last In, First Out (LIFO) — like a stack of plates. A queue is First In, First Out (FIFO) — like a line at a shop.

**Descriptive:** Stack examples: the method call stack, undo/redo, the browser's back button, checking matching brackets, DFS. Queue examples: print jobs, BFS, background job queues (like AWS SQS). In C#: `Stack<T>` (`Push`, `Pop`, `Peek`) and `Queue<T>` (`Enqueue`, `Dequeue`, `Peek`).

### Q89. How does a hash table (like `Dictionary`) work? What is a collision?

**One line:** It turns the key into a number (hash code), uses that number to pick a "bucket", and stores the item there — so finding it again is very fast (O(1) on average). A collision is when two keys land in the same bucket.

**Descriptive:** Collisions are handled by keeping a small list in each bucket (*chaining*) or by trying the next empty spot (*open addressing*). When the table gets too full, it grows and moves everything again. A bad `GetHashCode()` (for example, always returning the same number) puts everything in one bucket, and lookups become slow (O(n)). Don't change a key after putting it in a dictionary.

### Q90. What is a binary tree vs a binary search tree (BST)?

**One line:** In a binary tree, each node has at most two children. In a BST, smaller values always go left and bigger values always go right.

**Descriptive:** Searching, adding and removing in a BST takes O(log n) when the tree is balanced. But if you add items already in sorted order, the tree becomes one long line and gets slow (O(n)). Self-balancing trees (AVL, red-black) fix this — .NET's `SortedDictionary` uses a red-black tree. Ways to visit all nodes: in-order (gives sorted output in a BST), pre-order, post-order and level-order (BFS).

### Q91. What is a heap / priority queue?

**One line:** A heap is a tree where each parent is smaller than its children (min-heap) or bigger (max-heap). So you can always get the smallest (or biggest) item instantly, and adding or removing takes O(log n).

**Descriptive:** A heap is usually stored in an array. Uses: running tasks by priority, finding the shortest path (Dijkstra), finding the top 10 items. .NET 6+ has `PriorityQueue<TElement, TPriority>`.

### Q92. How do you represent a graph?

**One line:** With an adjacency list (each node keeps a list of its neighbors) or an adjacency matrix (a grid that marks which nodes are connected).

**Descriptive:** An adjacency list (`Dictionary<int, List<int>>`) uses less memory and is good when there are few connections — like social networks or road maps. A matrix uses much more memory (V × V) but can answer "are these two connected?" instantly — good when there are many connections. Graphs can be directed or not, and weighted or not. You explore them with BFS (using a queue) or DFS (using a stack or recursion).

### Q93. `Dictionary` vs `SortedDictionary` vs `SortedList` vs `HashSet`?

**One line:** `Dictionary` is fast and not sorted. `SortedDictionary` keeps keys sorted (a bit slower). `SortedList` is also sorted, uses less memory, but adding is slow. `HashSet` stores only unique values, with no keys.

**Descriptive:** Use `Dictionary` by default for finding things by key. Use `SortedDictionary` when you need to go through keys in order. Use `HashSet` to remove duplicates or to quickly check if something exists (`seen.Contains(x)`). It also has set operations like `UnionWith` and `IntersectWith`.

### Q94. Why do databases use B-trees for indexes rather than hash tables or binary trees?

**One line:** B-trees are short, wide and always balanced. They're designed for disk storage and can quickly find both one exact value and a range of values.

**Descriptive:** Each B-tree node holds many keys, so even millions of rows need only 3–4 levels, which means very few disk reads. The bottom level is linked in order, so queries like `WHERE Date BETWEEN ...` and `ORDER BY` are fast. A hash index can't do ranges — only exact matches. A binary tree would be too tall and need too many disk reads.

### Q95. What is recursion? What causes a stack overflow?

**One line:** Recursion is when a function calls itself with a smaller problem until it reaches a stopping point (base case). A stack overflow happens when it calls itself too many times or never stops.

**Descriptive:** Each call is added to the call stack. Example — factorial: `n <= 1 ? 1 : n * Fact(n - 1)`. If you forget the base case, it never stops and you get a `StackOverflowException`, which crashes the whole app in .NET. You can rewrite deep recursion using a loop and your own stack. Recursion works naturally for trees, DFS, merge sort and backtracking.

### Q96. How do you detect a cycle in a linked list?

**One line:** Use two pointers: a slow one that moves 1 step and a fast one that moves 2 steps. If they ever meet, there's a cycle (loop). This takes O(n) time and no extra memory.

**Descriptive:** If there's no cycle, the fast pointer reaches the end (null). Another way is to save every visited node in a `HashSet` — also O(n) time, but it uses extra memory. The same two-pointer trick finds the middle of a list: when the fast pointer reaches the end, the slow one is in the middle.

### Q97. Reverse a singly linked list.

**One line:** Walk through the list and turn each node's `Next` pointer to point backwards. It takes O(n) time and no extra memory.

```csharp
public ListNode? Reverse(ListNode? head)
{
    ListNode? prev = null;
    var current = head;
    while (current != null)
    {
        var next = current.Next; // remember the rest of the list
        current.Next = prev;     // turn the pointer backwards
        prev = current;          // move prev one step
        current = next;          // move current one step
    }
    return prev; // prev is the new head
}
```

Explain the three pointers out loud while you write — interviewers care more about your thinking than memorized code. Also mention the edge cases: an empty list and a list with one node.

## Algorithms & complexity (Q98–107)

### Q98. Linear search vs binary search?

**One line:** Linear search checks every item one by one — O(n) — and works on any list. Binary search cuts a *sorted* list in half at each step — O(log n).

```csharp
int BinarySearch(int[] a, int target)
{
    int lo = 0, hi = a.Length - 1;
    while (lo <= hi)
    {
        int mid = lo + (hi - lo) / 2; // safe way to find the middle
        if (a[mid] == target) return mid;
        if (a[mid] < target) lo = mid + 1; // look in the right half
        else hi = mid - 1;                 // look in the left half
    }
    return -1; // not found
}
```

For 1 million items, binary search needs only about 20 checks, while linear search may need 1 million. .NET has `Array.BinarySearch` and `List<T>.BinarySearch` built in.

### Q99. Compare common sorting algorithms. What does .NET use?

**One line:** Simple sorts (bubble, selection, insertion) are slow — O(n²). Better sorts (merge, quick, heap) are O(n log n). .NET's `Array.Sort` and `List.Sort` use introsort; LINQ's `OrderBy` keeps equal items in their original order (it's *stable*).

| Algorithm | Usual speed | Worst speed | Extra memory | Stable? |
| --- | --- | --- | --- | --- |
| Bubble sort | O(n²) | O(n²) | O(1) | Yes |
| Selection sort | O(n²) | O(n²) | O(1) | No |
| Insertion sort | O(n²) | O(n²) | O(1) | Yes |
| Merge sort | O(n log n) | O(n log n) | O(n) | Yes |
| Quick sort | O(n log n) | O(n²) | O(log n) | No |
| Heap sort | O(n log n) | O(n log n) | O(1) | No |

*Stable* means items with equal values stay in their original order — useful when you sort by one field and then by another. Insertion sort is actually fast for small or almost-sorted lists, so smart sorting methods use it for small parts.

### Q100. Merge sort vs quick sort?

**One line:** Both split the problem into smaller parts and are O(n log n). Merge sort is always O(n log n) and stable, but needs extra memory. Quick sort needs no extra array and is usually faster, but can become O(n²) if it picks bad pivots.

**Descriptive:** **Merge sort:** split the array in half, sort each half, then merge the two sorted halves together. Good for linked lists and very large files. **Quick sort:** pick a *pivot* value, move smaller items to the left and bigger items to the right, then repeat on each side. Picking a random pivot avoids the slow worst case on already-sorted data.

### Q101. BFS vs DFS?

**One line:** BFS (Breadth-First Search) explores level by level using a queue and finds the shortest path. DFS (Depth-First Search) goes as deep as possible first, using a stack or recursion.

**Descriptive:** Both take O(V + E) time (V = nodes, E = edges). Use BFS for the shortest path, "friends of friends" features, or printing a tree level by level. Use DFS for finding cycles, ordering tasks that depend on each other (topological sort), checking if a path exists, and puzzles like mazes. In graphs, keep a `visited` set so you don't go around in circles forever.

### Q102. Explain time vs space complexity, and best/average/worst case.

**One line:** Time complexity shows how running time grows as input grows. Space complexity shows how much extra memory is needed. Big-O usually describes the worst case.

**Descriptive:** Ignore constants and small terms: O(2n + 10) is just O(n). Quick sort is O(n log n) on average but O(n²) in the worst case. *Amortized* means the average cost over many operations — `List<T>.Add` sometimes needs to resize (slow), but on average it's O(1). There's often a trade-off: a `HashSet` uses O(n) extra memory to turn an O(n²) duplicate check into O(n).

### Q103. What are the two-pointer and sliding-window techniques?

**One line:** Two pointers move through the data from both ends or at different speeds. A sliding window is a moving range over the data. Both can turn slow O(n²) solutions into fast O(n) ones.

**Descriptive:** **Two pointers:** check a palindrome (one pointer from the start, one from the end, moving inward), or find two numbers that add up to a target in a *sorted* array. **Sliding window:** find the biggest sum of k items in a row (add the new item, remove the item that left the window), or the longest substring without repeated letters (grow the window on the right, shrink it on the left when a letter repeats, using a `HashSet` or `Dictionary`).

### Q104. What is dynamic programming? Explain memoization.

**One line:** Dynamic programming (DP) solves a problem by breaking it into smaller problems and saving their answers, so the same work isn't repeated. *Memoization* means saving the answers (usually in a dictionary).

**Descriptive:** Simple recursive Fibonacci is very slow (O(2ⁿ)) because it calculates the same values again and again. Saving results in a dictionary, or using a simple loop with two variables, makes it O(n). Classic DP problems: climbing stairs, coin change, longest common subsequence, knapsack. Think of DP when a problem asks for a minimum, maximum or count, and each step depends on earlier results.

### Q105. How do you check if a string is a palindrome, or if two strings are anagrams?

**One line:** Palindrome: compare letters from both ends moving to the middle — O(n). Anagram: count each letter in both strings and compare the counts — O(n) — or sort both strings and compare — O(n log n).

```csharp
bool IsPalindrome(string s)
{
    int i = 0, j = s.Length - 1;
    while (i < j)
        if (s[i++] != s[j--]) return false;
    return true;
}
```

Ask questions first: Does upper/lower case matter? Should spaces and punctuation be ignored? Interviewers like it when you check your assumptions.

### Q106. Find duplicates in an array, or the first non-repeating character in a string.

**One line:** Use a `HashSet` to find duplicates in one pass — O(n). Use a `Dictionary<char, int>` to count each character, then go through the string again to find the first one with count 1 — O(n).

**Descriptive:** Two loops would be O(n²); sorting first would be O(n log n). The hash-based way uses more memory but is the fastest — O(n). With LINQ: `nums.GroupBy(x => x).Where(g => g.Count() > 1).Select(g => g.Key)`. It's easy to read, but mention that a normal loop can be faster because it can stop early.

### Q107. How do you approach a coding problem in an interview?

**One line:** Understand the problem, try an example, explain a simple (brute-force) solution, improve it, write clean code, and test it with edge cases — talking out loud the whole time.

**Descriptive:**

1. Repeat the problem in your own words. Ask about input size, limits and edge cases (empty, null, negative numbers, duplicates).
2. Solve a small example by hand.
3. Explain the simple brute-force solution and its Big-O.
4. Look for a better way — hashing, sorting, two pointers, BFS/DFS, DP.
5. Write clean code with clear names.
6. Walk through your code with the example and edge cases, and say the final time and space complexity.

## SQL & relational design fundamentals (Q108–125)

### Q108. What is a relational database? How does it differ from NoSQL?

**One line:** A relational database stores data in tables (rows and columns) that are linked by keys. It has a fixed structure, uses SQL, and supports transactions. NoSQL databases store data in more flexible ways (documents, key-value pairs, graphs).

**Descriptive:** Relational databases (SQL Server, PostgreSQL) are best for structured, connected data that must stay correct — orders, invoices, users. NoSQL databases (MongoDB, DynamoDB, Redis) are good for flexible data, very high traffic, or simple key-based lookups. Many SaaS apps use both: PostgreSQL as the main database and Redis as a cache.

### Q109. What are DDL, DML, DCL and TCL?

**One line:** DDL defines the structure (`CREATE`, `ALTER`, `DROP`, `TRUNCATE`). DML works with the data (`SELECT`, `INSERT`, `UPDATE`, `DELETE`). DCL controls permissions (`GRANT`, `REVOKE`). TCL controls transactions (`COMMIT`, `ROLLBACK`, `SAVEPOINT`).

### Q110. Explain candidate, composite, surrogate and natural keys.

**One line:** A candidate key is any column (or columns) that could uniquely identify a row; the one you choose becomes the primary key. A composite key uses two or more columns. A natural key has real-world meaning (like email). A surrogate key has no meaning — it's just an ID number or GUID.

**Descriptive:** Surrogate keys are usually better as primary keys, because real-world values can change (people change emails) and can be long. Still, add a `UNIQUE` constraint on natural keys like email. Composite keys are common in linking tables, for example `(StudentId, CourseId)`.

### Q111. How do you model one-to-one, one-to-many and many-to-many relationships?

**One line:** One-to-many: put a foreign key on the "many" side. One-to-one: a foreign key that must be unique. Many-to-many: add a middle (junction) table with two foreign keys.

**Descriptive:**

- **One-to-many:** one customer has many orders → `Orders.CustomerId`.
- **One-to-one:** one user has one profile → `UserProfiles.UserId` (unique).
- **Many-to-many:** students take many courses and courses have many students → `Enrollments(StudentId, CourseId, EnrolledAt, Grade)`.

The middle table can also store extra details about the relationship, like the enrollment date or grade.

### Q112. Design a simple database schema for an online store.

**One line:** Main tables: `Customers`, `Products`, `Categories`, `Orders`, `OrderItems` (which product, how many, at what price) and `Payments`.

**Descriptive:**

- `Customers(Id PK, Name, Email UNIQUE, CreatedAt)`
- `Categories(Id PK, Name)`; `Products(Id PK, CategoryId FK, Name, Price DECIMAL(18,2), Stock)`
- `Orders(Id PK, CustomerId FK, Status, OrderDate, TotalAmount)`
- `OrderItems(OrderId FK, ProductId FK, Quantity, UnitPrice, PK(OrderId, ProductId))`
- `Payments(Id PK, OrderId FK, Amount, Method, PaidAt)`

Important point: save `UnitPrice` in `OrderItems`. Product prices change later, but old orders must keep the price the customer actually paid. Also add indexes on foreign keys and on columns you often search, like `Orders.Status` and `OrderDate`.

### Q113. What are aggregate functions? `COUNT(*)` vs `COUNT(column)`?

**One line:** Aggregate functions (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) calculate one result from many rows, usually with `GROUP BY`. `COUNT(*)` counts all rows; `COUNT(column)` skips rows where that column is NULL.

**Descriptive:** `COUNT(DISTINCT CustomerId)` counts unique customers. Every column in `SELECT` that isn't inside an aggregate must be in `GROUP BY`. `AVG` also skips NULLs, which can surprise you — `AVG(COALESCE(Score, 0))` counts missing scores as zero.

### Q114. Subquery vs JOIN — and `EXISTS` vs `IN`?

**One line:** A JOIN combines columns from several tables. A subquery is a query inside another query. `EXISTS` stops as soon as it finds one match; `IN` checks if a value is in a list.

**Descriptive:** A *correlated* subquery uses a value from the outer query, so it runs once per row. Example — employees who earn more than their department's average: `WHERE Salary > (SELECT AVG(Salary) FROM Employees e2 WHERE e2.DeptId = e1.DeptId)`. Modern databases often run `IN`, `EXISTS` and JOIN versions the same way. But `EXISTS` clearly means "has at least one" and is safe with NULLs. Write the clearest version, then check the execution plan.

### Q115. How does NULL behave in SQL?

**One line:** NULL means "unknown". So `= NULL` never works — use `IS NULL` or `IS NOT NULL`. Use `COALESCE` to give a default value.

**Descriptive:** `WHERE Phone = NULL` returns nothing; write `WHERE Phone IS NULL`. `COALESCE(Phone, 'N/A')` returns the first value that isn't NULL (SQL Server also has `ISNULL`). A common trap: `WHERE Id NOT IN (SELECT ManagerId FROM Employees)` returns *no rows at all* if any `ManagerId` is NULL. Use `NOT EXISTS` instead. Also, NULL + anything = NULL in math and text joining.

### Q116. What are window functions? `ROW_NUMBER` vs `RANK` vs `DENSE_RANK`?

**One line:** Window functions calculate values across related rows but still keep every row (unlike `GROUP BY`). With ties: `ROW_NUMBER` gives 1, 2, 3; `RANK` gives 1, 1, 3; `DENSE_RANK` gives 1, 1, 2.

**Descriptive:** Example — the highest-paid employee in each department:

```sql
SELECT *
FROM (
    SELECT e.*, DENSE_RANK() OVER (PARTITION BY DepartmentId ORDER BY Salary DESC) AS rnk
    FROM Employees e
) t
WHERE rnk = 1;
```

`PARTITION BY` restarts the ranking for each department. Other useful ones: running totals with `SUM(Amount) OVER (ORDER BY OrderDate)`, and `LAG`/`LEAD` to compare a row with the previous or next row (e.g. this month vs last month).

### Q117. What is a CTE? What is a recursive CTE used for?

**One line:** A CTE (`WITH name AS (...)`) is a named, temporary result you can use inside one query. It makes long SQL easier to read. A recursive CTE calls itself to walk through tree-like data.

**Descriptive:** CTEs replace deeply nested subqueries with clear steps. Recursive CTEs are used for tree data stored in a table, like a company org chart (`Employees.ManagerId`) or categories inside categories. The first part picks the top level, and the second part keeps adding the next level until there are no more rows.

### Q118. How do you find and delete duplicate rows?

**One line:** Find them with `GROUP BY ... HAVING COUNT(*) > 1`. Delete them by numbering duplicates with `ROW_NUMBER()` and removing every copy except the first.

```sql
WITH d AS (
    SELECT Id, ROW_NUMBER() OVER (PARTITION BY Email ORDER BY Id) AS rn
    FROM Customers
)
DELETE FROM Customers WHERE Id IN (SELECT Id FROM d WHERE rn > 1);
```

This works in both SQL Server and PostgreSQL. Afterwards, add a `UNIQUE` constraint so duplicates can't come back. Always test deletes inside a transaction or on a backup first.

### Q119. `CHAR` vs `VARCHAR` vs `NVARCHAR`? Which type for money?

**One line:** `CHAR(n)` always uses a fixed length (fills with spaces). `VARCHAR(n)` uses only the length it needs. `NVARCHAR` (SQL Server) supports Unicode — needed for Bangla and other languages. For money, use `DECIMAL`/`NUMERIC`, never `FLOAT`.

**Descriptive:** `FLOAT` stores approximate values, so 0.1 + 0.2 isn't exactly 0.3. Small rounding errors in money are not acceptable. In C#, likewise, use `decimal` (not `double`) for money. Store dates as `DATETIME2`/`DATETIMEOFFSET` (SQL Server) or `timestamptz` (PostgreSQL), preferably in UTC. Use the smallest type that fits — it saves space and makes indexes faster.

### Q120. What are views, stored procedures and triggers? Pros and cons?

**One line:** A view is a saved query. A stored procedure is saved SQL code you call with parameters. A trigger runs automatically when rows are inserted, updated or deleted.

**Descriptive:** **Views** make complex joins simple to reuse and can hide columns users shouldn't see. **Stored procedures** reduce trips between the app and database, use parameters (safe from SQL injection), and can have their own permissions — but business logic inside the database is harder to test, version and debug than C# code. **Triggers** are useful for audit logs, but they're "hidden" logic that can surprise developers and slow things down, so use them carefully.

### Q121. What is SQL injection and how do you prevent it?

**One line:** SQL injection is when an attacker types SQL code into an input box, and the app adds it directly into a query. Prevent it with parameterized queries (or EF Core) — never build SQL by joining strings.

**Descriptive:** With `"SELECT * FROM Users WHERE Name = '" + input + "'"`, an input like `' OR '1'='1` returns every user. Worse input could even delete tables. Parameters send the value separately from the SQL, so the database never runs it as code. Use `cmd.Parameters.AddWithValue("@name", input)`, EF Core LINQ, or `FromSqlInterpolated`. Also give the app's database account only the permissions it needs, and validate input.

### Q122. Auto-increment integer vs GUID primary keys?

**One line:** Integers are small, fast and in order. GUIDs are unique everywhere (you can create them anywhere, and they're hard to guess), but they're bigger and random ones can slow down inserts.

**Descriptive:** Random GUIDs (`NEWID()`) inserted into a clustered index land in random places, which causes fragmentation and slower inserts. Sequential GUIDs (`NEWSEQUENTIALID()` or UUIDv7) reduce this. If you use simple numbers in URLs (`/orders/5`), users can guess other IDs (`/orders/6`) — so always check permissions, whatever key type you use. A common mix: an integer ID inside the database plus a GUID for public use.

### Q123. Write a query: top 3 customers by total spending in the last 30 days.

**One line:** Join customers and orders, keep only the last 30 days, group by customer, add up their totals, sort from highest, and take 3.

```sql
-- SQL Server
SELECT TOP 3 c.Id, c.Name, SUM(o.TotalAmount) AS TotalSpent
FROM Customers c
JOIN Orders o ON o.CustomerId = c.Id
WHERE o.OrderDate >= DATEADD(DAY, -30, SYSUTCDATETIME())
GROUP BY c.Id, c.Name
ORDER BY TotalSpent DESC;

-- PostgreSQL
SELECT c.id, c.name, SUM(o.total_amount) AS total_spent
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE o.order_date >= NOW() - INTERVAL '30 days'
GROUP BY c.id, c.name
ORDER BY total_spent DESC
LIMIT 3;
```

Also mention: what if two customers tie (use `TOP 3 WITH TIES` or `DENSE_RANK`)? And should cancelled orders be excluded?

### Q124. List each employee with their manager's name.

**One line:** Join the `Employees` table with itself (a self join). Use a LEFT JOIN so employees with no manager (like the CEO) still appear.

```sql
SELECT e.Name AS Employee, m.Name AS Manager
FROM Employees e
LEFT JOIN Employees m ON m.Id = e.ManagerId;
```

### Q125. Soft delete vs hard delete? What are audit columns?

**One line:** A hard delete removes the row forever. A soft delete only marks it as deleted (with `IsDeleted` or `DeletedAt`), so it can be restored and history is kept. Audit columns record who changed what and when.

**Descriptive:** Soft delete is common in SaaS apps for recovery and legal reasons. But every query must skip deleted rows — EF Core's *global query filters* (`HasQueryFilter(x => !x.IsDeleted)`) do this automatically. Be careful with unique rules: a deleted user's email might block someone from signing up again (fix it with a filtered unique index). Common audit columns: `CreatedAt`, `CreatedBy`, `UpdatedAt`, `UpdatedBy`. They're often filled in automatically by overriding `SaveChanges`.

## REST & HTTP fundamentals (Q126–135)

### Q126. What does an HTTP request and response consist of?

**One line:** A request has a method, a URL, headers and sometimes a body. A response has a status code, headers and sometimes a body.

**Descriptive:** Example request: `POST /api/orders HTTP/1.1`, with headers `Content-Type: application/json` and `Authorization: Bearer ...`, and a body `{ "productId": 5, "quantity": 2 }`. Example response: `HTTP/1.1 201 Created`, with headers `Location: /api/orders/42` and `Content-Type: application/json`, and a body with the new order.

### Q127. Which HTTP headers should a backend developer know?

**One line:** `Content-Type`, `Accept`, `Authorization`, `Cache-Control`, `ETag`/`If-None-Match`, `Location`, `Set-Cookie`/`Cookie`, and CORS headers like `Access-Control-Allow-Origin`.

**Descriptive:**

- `Content-Type` — what format the body is in (e.g. JSON).
- `Accept` — what format the client wants back.
- `Authorization` — login proof, like a Bearer token.
- `Location` — the URL of a newly created item.
- `Cache-Control` / `ETag` — caching rules.

Teams also add custom headers like `X-Correlation-ID` to follow one request across services.

### Q128. HTTP vs HTTPS?

**One line:** HTTPS is HTTP with encryption (TLS). It hides the data from others, proves the server is real (using a certificate), and stops data from being changed on the way.

**Descriptive:** Without HTTPS, anyone on the same network (like public Wi-Fi) could read passwords and tokens, or change the responses. In the TLS handshake, the browser checks the server's certificate (from a trusted authority), and then both sides agree on secret keys for encryption. In ASP.NET Core, `UseHttpsRedirection()` and HSTS force HTTPS. On AWS, the load balancer often handles HTTPS.

### Q129. Path parameters vs query parameters vs request body — when to use each?

**One line:** Path parameters pick one specific item (`/orders/42`). Query parameters filter, sort or page (`?status=paid&page=2`). The body carries the data for creating or updating (POST/PUT/PATCH).

**Descriptive:** Don't put sensitive data (like passwords) in URLs — URLs get saved in logs and browser history. GET requests shouldn't have a body. In ASP.NET Core, these map to `[FromRoute]`, `[FromQuery]` and `[FromBody]`, and headers use `[FromHeader]`.

### Q130. If REST is stateless, how does the server know who the user is?

**One line:** Every request carries its own proof of identity — usually a JWT in the `Authorization` header, or a session cookie. So the server doesn't need to remember earlier requests.

**Descriptive:** With a JWT, the user's ID and role are inside the signed token itself. With cookie sessions, the cookie holds a session ID, and the session data is kept in a shared store like Redis — so any server can handle the request. Never keep session data in one server's memory when you have several servers behind a load balancer. Cookies should be `HttpOnly`, `Secure` and `SameSite` to make attacks harder.

### Q131. How should an API return errors?

**One line:** Use the right HTTP status code plus a consistent error body. In ASP.NET Core, use the standard `ProblemDetails` format.

**Descriptive:** `ProblemDetails` has fields like `type`, `title`, `status`, `detail` and `instance`. `ValidationProblemDetails` adds an `errors` list for each field. This way, the frontend can handle all errors the same way. Never show stack traces or SQL errors to users in production. Log them on the server with a correlation ID, and return that ID so support can find the error.

### Q132. What is rate limiting and how do you implement it?

**One line:** Rate limiting sets a maximum number of requests a client can make in a time period. It protects the API from abuse and overload. Extra requests get 429 Too Many Requests.

**Descriptive:** Common methods: fixed window, sliding window, token bucket, and limiting how many requests run at once. ASP.NET Core (from .NET 7) has this built in: `builder.Services.AddRateLimiter(...)` and `app.UseRateLimiter()`. Limits are usually per user, per API key or per IP address. With several servers, keep the counters in Redis so all servers share the same count. API gateways (like AWS API Gateway) can also do it.

### Q133. How do you document and test an API manually?

**One line:** Document it with Swagger/OpenAPI (generated from your code), and test it by hand with Swagger UI, Postman or `.http` files.

**Descriptive:** Swagger reads your controllers and DTOs and creates a web page listing every endpoint. The frontend and QA teams use it as the agreement for how the API works. Postman lets you save requests, switch between environments (local, staging) and add automatic checks. Good docs show every endpoint, example requests and responses, status codes, and what login is needed.

### Q134. REST vs SOAP vs GraphQL vs gRPC?

**One line:** REST uses URLs and HTTP methods with JSON. SOAP is an older, strict XML-based style. GraphQL uses one endpoint where the client asks for exactly the fields it needs. gRPC uses a fast binary format, mainly for services talking to each other.

**Descriptive:** REST is the default for most web and mobile APIs because it's simple and easy to cache. GraphQL helps complex screens avoid getting too much or too little data, but caching and permissions are harder. gRPC is very fast for internal microservices. SOAP is still found in older banking and enterprise systems.

### Q135. How does HTTP caching work? What are ETags?

**One line:** `Cache-Control` tells browsers and proxies if and how long they can keep a response. An ETag is a version tag — the client can ask "has this changed?" and get a quick 304 Not Modified if it hasn't.

**Descriptive:** `Cache-Control: public, max-age=3600` means "you can cache this for 1 hour". `no-store` means "never cache this" — use it for private data. The client sends `If-None-Match: "abc123"`; if the data hasn't changed, the server replies 304 with no body, which saves bandwidth. ETags can also stop two people overwriting each other's changes, using `If-Match` — if the version doesn't match, the server returns 412 Precondition Failed. ASP.NET Core has built-in caching middleware.

## .NET & ASP.NET Core quick hits (Q136–143)

### Q136. What is Kestrel?

**One line:** Kestrel is the web server built into ASP.NET Core. It's fast, works on any OS, and is what actually receives the HTTP requests.

**Descriptive:** Kestrel can face the internet directly, but in production it often sits behind a reverse proxy or load balancer (like Nginx, IIS or AWS ALB). That front layer handles HTTPS, spreads traffic across servers, and serves static files. When behind a proxy, turn on the Forwarded Headers middleware so the app sees the user's real IP address.

### Q137. What happens in `Program.cs` in a modern ASP.NET Core app?

**One line:** It creates the app builder, registers services, builds the app, sets up the middleware pipeline, maps the endpoints, and starts the app.

**Descriptive:** There are two parts:

1. **Register services** with `builder.Services.Add...()` — DbContext, authentication, controllers, Swagger, Redis.
2. **After `var app = builder.Build();`** — add middleware with `app.Use...()`, map endpoints with `app.MapControllers()`, and finally call `app.Run()`.

Older projects split this into a `Startup` class with `ConfigureServices` and `Configure` methods.

### Q138. Filters vs middleware?

**One line:** Middleware runs for every request, at the start of the pipeline. Filters run only around controller actions, and they can see action details like the arguments and model validation.

**Descriptive:** Filter types, in the order they run: authorization, resource, action (before and after the action method), exception and result filters. Use middleware for things that apply to everything (logging, CORS, global error handling). Use filters for controller-specific things (checking a model, auditing an action). Filters can be added globally, to one controller, or to one action using attributes.

### Q139. `IActionResult` vs `ActionResult<T>`?

**One line:** `IActionResult` can return any result, but doesn't say what type of data comes back. `ActionResult<T>` returns either a `T` or a status result, and tools like Swagger know the response type.

**Descriptive:** `public async Task<ActionResult<OrderDto>> Get(int id)` can `return order;` (200 OK) or `return NotFound();`. Helper methods: `Ok()`, `Created()`/`CreatedAtAction()`, `NoContent()`, `BadRequest()`, `NotFound()`, `Conflict()`, `Unauthorized()`, `Forbid()`. Minimal APIs use `Results.Ok(...)` or `TypedResults`.

### Q140. How should you use `HttpClient` correctly?

**One line:** Don't create a new `HttpClient` for every request. Use `IHttpClientFactory` instead, so connections are reused safely.

**Descriptive:** Creating and disposing `new HttpClient()` for every call can use up all available network connections (sockets) when the app is busy. Using one static `HttpClient` forever can miss DNS changes. `builder.Services.AddHttpClient<IPaymentGateway, PaymentGateway>(c => c.BaseAddress = ...)` solves both problems. For calls to outside services, also add timeouts and automatic retries (with Polly or `Microsoft.Extensions.Http.Resilience`).

### Q141. How do you run background work in ASP.NET Core?

**One line:** Create a class that inherits `BackgroundService` (or implements `IHostedService`) for long-running work. For important jobs that must not be lost, use a queue (AWS SQS) or a library like Hangfire.

**Descriptive:** Don't start background work with `Task.Run` inside a controller and forget about it — if the app restarts, the work is lost, and errors go unnoticed. A `BackgroundService` runs `ExecuteAsync` and gets a cancellation token so it can stop cleanly. It's a singleton, so to use scoped services like `DbContext`, create a scope with `IServiceScopeFactory`. Common jobs: sending emails, making reports, deleting expired tokens.

### Q142. What is a DTO and why not return EF entities directly from the API?

**One line:** A DTO (Data Transfer Object) is a simple class shaped for what the API sends or receives. It keeps the API separate from the database design and hides private fields.

**Descriptive:** Returning database entities directly can:

- Leak private data, like `PasswordHash`.
- Cause endless loops when converting to JSON (Order → Customer → Orders → ...).
- Trigger extra database queries while converting to JSON.
- Break the API every time the database changes.

Accepting entities as input also lets users change fields they shouldn't (over-posting). Convert entities to DTOs using LINQ `Select`, simple mapping methods, or libraries like Mapster or AutoMapper.

### Q143. How would you structure the layers of a .NET Web API project?

**One line:** Controllers handle HTTP. Services hold the business logic. Repositories or `DbContext` handle the database. Each layer depends on interfaces, connected through DI.

**Descriptive:** A common structure:

- **API layer** — controllers, middleware, DTOs.
- **Application/Service layer** — business rules and validation.
- **Domain layer** — entities and core logic.
- **Infrastructure layer** — EF Core, Redis, email, AWS code.

In *Clean Architecture*, the business logic doesn't depend on any framework or database. Keep controllers thin: receive the request, call a service, return the result. This separation makes the services easy to unit test.

## Git & tooling fundamentals (Q144–147)

### Q144. What is Git, and how is it different from GitHub?

**One line:** Git is a tool that tracks changes to your code on your own computer. GitHub (like GitLab or Bitbucket) is a website that stores Git projects online and adds pull requests, code reviews, issues and CI.

**Descriptive:** Git is *distributed* — every copy of the project has the full history, so you can commit, make branches and see history even without internet. Git saves your project as a series of snapshots called commits. Each commit has a unique ID (a hash) and points to the one before it.

### Q145. Explain the working directory, staging area and repository — and `fetch` vs `pull`.

**One line:** You change files in the working directory, `git add` puts changes in the staging area, and `git commit` saves them into the repository. `fetch` downloads new changes without merging them; `pull` downloads *and* merges them.

**Descriptive:** The staging area lets you choose exactly which changes go into the next commit. `git status` shows what's changed and what's staged. `git diff --staged` shows exactly what will be committed. Use `git fetch` first if you want to look at incoming changes safely (`git log origin/main`) before merging them. `git pull --rebase` keeps the history in a straight line.

### Q146. What is `.gitignore`, and what should never be committed?

**One line:** `.gitignore` is a file that lists files Git should ignore. Never commit build output, IDE settings, downloaded packages or secrets.

**Descriptive:** For .NET, ignore: `bin/`, `obj/`, `.vs/`, `*.user`, `TestResults/`, `.env` files, and any local `appsettings.Development.json` that has secrets. Run `dotnet new gitignore` to create a standard one. If a file was already committed, adding it to `.gitignore` isn't enough — remove it with `git rm --cached`. And if a secret was exposed, change it immediately.

### Q147. What branching strategies do teams use? What makes a good pull request?

**One line:** Two common strategies: trunk-based (small branches merged into `main` often) and Git Flow (separate `develop`, `feature`, `release` and `hotfix` branches). A good pull request is small, focused, clearly described and tested.

**Descriptive:** Trunk-based development works well with CI/CD. Git Flow suits teams that release on a fixed schedule. Releases are marked with tags, like `v1.4.0`. A good PR has:

- A clear title.
- A description of *what* changed and *why*.
- A link to the task or ticket.
- Screenshots or sample requests, if useful.
- Passing tests and CI.
- A small size (ideally a few hundred lines at most), so reviewers can check it properly.

## Attitude & learning questions (Q148–150)

The circular clearly asks for "willingness to learn, ask questions, and take feedback", so expect at least one of these. Answer with a short, real example from a university project, thesis or internship.

### Q148. Why do you want to work on backend development for a SaaS product?

**One line:** I enjoy building APIs and designing databases that many people depend on, and working on a real SaaS product is the best way to learn how professional software is built, released and monitored.

**Descriptive:** Connect your answer to the job: building REST APIs with C# and .NET, writing good SQL, and learning deployment and monitoring with AWS and Jenkins. Mention something you really enjoyed in your own project (for example, speeding up a slow query or designing the database). Show that you understand what SaaS needs: many customers using the same system, reliability, security, and releasing new features often.

### Q149. Tell me about a time you received critical feedback. How did you respond?

**One line:** Use the STAR method — Situation, Task, Action, Result — and show that you listened, improved, and kept using what you learned.

**Descriptive:** Example: "In my final-year project, my supervisor said my controller had all the business logic and SQL in one place (Situation/Task). At first I didn't fully understand why that was a problem, so I asked for an example. Then I read about layered architecture and split the code into services and repositories with interfaces (Action). After that, writing unit tests became much easier, and now I structure every project this way (Result)." Don't choose a story where you got defensive or blamed someone else.

### Q150. How do you learn a new technology quickly — for example, one listed here that you haven't used yet?

**One line:** I start with the official documentation, build a small practice project, compare it with something I already know, then ask experienced teammates and use it in a real task.

**Descriptive:** Be honest about what you haven't used — for example, "I haven't used Jenkins in a real job yet." Then show how you learn:

1. Read the official docs and a getting-started guide (Microsoft Learn for .NET, the AWS free tier for AWS).
2. Build a tiny demo — for example, a Redis cache in a sample API.
3. Compare it with tools you know — for example, Jenkins pipelines vs GitHub Actions.
4. Look at how the team already uses it in the real codebase.

Give one example of something you taught yourself recently and how long it took — a real example is more convincing than just saying "I learn fast".
