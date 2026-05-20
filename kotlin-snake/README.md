# kotlin-hello-world

A minimal Kotlin + Swing Snake game built with Gradle. The goal is not to be impressive — it's to be readable. Every file here exists for a reason, and this document explains what that reason is.

---

## Table of Contents

1. [What is this?](#what-is-this)
2. [Project Structure](#project-structure)
3. [How Kotlin and Gradle Work Together](#how-kotlin-and-gradle-work-together)
4. [Common Commands](#common-commands)
5. [Adding Dependencies](#adding-dependencies)
6. [How the Build Pipeline Works](#how-the-build-pipeline-works)

---

## What is this?

A minimal Kotlin + Swing game — Snake, running in a 600×600 window on a 30×30 grid. The point is to show that you can build a real, interactive game with nothing but the JDK. No game engines, no frameworks, no external libraries. Just Kotlin, Gradle, and classes that have shipped with Java since the 90s.

The educational angle is twofold: it's a clean example of how Kotlin and Gradle fit together, and it demonstrates what Swing gives you out of the box — a window, a game loop via `javax.swing.Timer`, keyboard input via `KeyListener`, and custom rendering via `paintComponent`.

---

## Project Structure

```
kotlin-hello-world/
├── gradlew                        # Gradle wrapper script (Unix/macOS)
├── gradlew.bat                    # Gradle wrapper script (Windows)
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar     # Bootstrapper binary
│       └── gradle-wrapper.properties  # Which Gradle version to use
├── gradle.properties              # Gradle runtime config flags
├── settings.gradle.kts            # Project name + subproject declarations
├── app/
│   ├── build.gradle.kts           # Build instructions for the app subproject
│   └── src/
│       └── main/
│           └── kotlin/
│               └── App.kt         # The application source
```

### `gradlew` and `gradlew.bat`

These are the **Gradle Wrapper** scripts. Instead of requiring everyone on the team (or a CI server) to have Gradle pre-installed, the wrapper downloads and runs the exact version of Gradle your project specifies.

You run `./gradlew <task>` instead of `gradle <task>`. This matters because:

- **Version locking** — the project always uses the Gradle version it was built with, regardless of what anyone has installed globally.
- **No manual install** — a fresh checkout just works.
- **CI-friendly** — CI runners don't need Gradle pre-configured.

The wrapper delegates to the real Gradle distribution, which it downloads on first use and caches in `~/.gradle/wrapper/dists/`.

### `gradle/wrapper/`

This directory contains two files:

- **`gradle-wrapper.jar`** — a small binary that knows how to download Gradle. It's committed to the repo so the wrapper can bootstrap itself with no external dependencies.
- **`gradle-wrapper.properties`** — tells the wrapper which Gradle version and distribution URL to use. This project uses Gradle 9.5.1. When you change the Gradle version (via `./gradlew wrapper --gradle-version X.Y.Z`), this file is what gets updated.

You should commit both of these files.

### `gradle.properties`

```properties
org.gradle.configuration-cache=true
```

This is a key-value config file that Gradle reads at startup. Right now it enables one flag:

**Configuration cache** (`org.gradle.configuration-cache=true`) — Gradle can serialize the result of the configuration phase (reading build scripts, resolving task graphs) and reuse it on subsequent builds if nothing relevant changed. This makes repeated builds noticeably faster because Gradle skips re-running your `build.gradle.kts` files from scratch every time.

### `settings.gradle.kts`

```kotlin
rootProject.name = "kotlin-hello-world"
include("app")
```

This file defines the **project name** and declares **subprojects**.

Gradle supports two build styles:

- **Single-project** — one `build.gradle.kts` at the root, everything lives there. Simple, but doesn't scale well.
- **Multi-project** — a root project that coordinates several subprojects, each with their own `build.gradle.kts`. Common in real applications (e.g., separate modules for `app`, `core`, `api`).

This project uses a multi-project layout even though there's only one subproject (`app`). It's a good habit — it means adding a second module later doesn't require restructuring.

The `include("app")` line tells Gradle: "there's a subdirectory called `app/` that is a subproject with its own build file."

### `app/build.gradle.kts`

This is the main build script for the `app` subproject. Let's go through each block.

```kotlin
plugins {
    kotlin("jvm") version "2.2.0"
    application
}
```

**Plugins** extend what Gradle can do. Without plugins, Gradle has no idea how to compile Kotlin.

- `kotlin("jvm")` — applies the Kotlin JVM plugin. This tells Gradle how to find `.kt` files, invoke the Kotlin compiler, and produce JVM bytecode. It also sets up source set conventions (i.e., "look for source files in `src/main/kotlin`").
- `application` — a Gradle built-in plugin that adds a `run` task and lets you specify a main class. Without this, `./gradlew run` wouldn't exist.

```kotlin
repositories {
    mavenCentral()
}
```

**Repositories** are where Gradle goes to download dependencies. `mavenCentral()` refers to the Maven Central repository — the largest public repository of JVM libraries. When you declare a dependency, Gradle fetches it from here (and caches it in `~/.gradle/caches/`).

```kotlin
kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21
    }
}
```

The Kotlin compiler produces JVM bytecode, and JVM bytecode has versions. `jvmTarget = JVM_21` tells the Kotlin compiler: "produce bytecode compatible with Java 21." This should match the Java version you're running. If you set this too high and run on an older JVM, you'll get an `UnsupportedClassVersionError` at runtime.

```kotlin
tasks.withType<JavaCompile> {
    targetCompatibility = "21"
}
```

This sets the same target for any Java compilation tasks — relevant if you ever mix Kotlin and Java source files in the same project.

```kotlin
dependencies {
    testImplementation("org.jetbrains.kotlin:kotlin-test")
}
```

**Dependencies** are external libraries your code needs. The string format is `"group:artifact:version"`. Here, `kotlin-test` is pulled in for tests only (see [Adding Dependencies](#adding-dependencies) for the difference between scopes).

```kotlin
application {
    mainClass = "AppKt"
}
```

This tells the `application` plugin which class contains the entry point. See [How Kotlin and Gradle Work Together](#how-kotlin-and-gradle-work-together) for why the class is named `AppKt` and not `App`.

### `app/src/main/kotlin/App.kt`

This is the entire game — about 230 lines split into two classes and a `main` function.

```kotlin
class SnakeGame { ... }   // pure data + logic (no UI)
class GamePanel { ... }   // JPanel subclass — rendering + input
fun main() { ... }        // creates the JFrame window and starts the game
```

**`SnakeGame`** holds all game state: the snake's body as a list of grid coordinates, the current direction, the food position, score, and whether the game is over. It has no knowledge of Swing — it's plain Kotlin data and logic. This separation makes it easy to reason about: the game state is just a data structure that gets updated on each tick.

**`GamePanel`** is a `JPanel` subclass that handles two things:

- **Rendering** — `paintComponent` is called by Swing whenever the panel needs to be redrawn. It draws the grid, snake segments (rounded rectangles), food (with a shine dot), eyes on the snake's head, and the score overlay.
- **Input** — `KeyListener` captures arrow keys and WASD for direction changes, and R/Space for restarting after game over.

**The game loop** is driven by a `javax.swing.Timer` that fires every 120ms. On each tick it updates the game state and calls `repaint()`, which schedules a call to `paintComponent`. This is the standard Swing pattern for animation.

A few Kotlin conventions worth noting:

- **Top-level `fun main()`** — unlike Java, you don't need a class to hold the entry point. It sits directly in the file.
- **Classes without `public`** — Kotlin's default visibility is `public`, so the keyword is omitted.
- **No package declaration** — for a single-file project like this, it's fine. In larger projects you'd add `package com.example.app` to avoid naming collisions.

### A note on Swing and dependencies

The game uses `java.awt.*` and `javax.swing.*` — both part of the `java.desktop` module in the JDK. They've shipped with Java since the late 90s and require zero additional dependencies. Nothing to add to `build.gradle.kts`, nothing to download from Maven Central.

This is worth appreciating: a windowed, interactive, animated application with no dependencies beyond the JDK itself. Swing isn't fashionable, but it's capable and available everywhere Java runs.

---

## How Kotlin and Gradle Work Together

Here's the mental model:

```
Your .kt source files
        |
        v
  Gradle invokes the Kotlin compiler (kotlinc)
        |
        v
  Kotlin compiler produces .class files (JVM bytecode)
        |
        v
  JVM runs the bytecode
```

Gradle itself is the **orchestrator** — it reads your build scripts, resolves dependencies, and decides which tasks to run and in what order. The Kotlin plugin is what teaches Gradle how to do the Kotlin-specific parts.

### Why is the main class `AppKt` and not `App`?

Kotlin compiles each file that contains top-level declarations (functions, properties) into a JVM class. For `App.kt`, that class is named `AppKt` — Kotlin appends `Kt` to the filename to avoid collisions with any class you might also define in that file named `App`.

So when you write:

```kotlin
// App.kt
fun main() { ... }
```

The Kotlin compiler produces a class `AppKt` with a static `main` method. The JVM entry point needs a class name, which is why `mainClass = "AppKt"` — not `"App"`.

If your file were named `Main.kt`, the generated class would be `MainKt`, and you'd write `mainClass = "MainKt"`.

### Configuration cache and incremental compilation

Two mechanisms keep builds fast:

- **Configuration cache** — Gradle caches the task graph from your build scripts. On subsequent builds where nothing in the build scripts changed, Gradle skips the configuration phase entirely.
- **Incremental compilation** — The Kotlin compiler tracks which source files changed and only recompiles affected files, not the entire project.

Together, these make the feedback loop fast once the initial build is done.

---

## Common Commands

All commands use `./gradlew` (the wrapper). On Windows, use `gradlew.bat` instead.

| Command                  | What it does                                                         |
| ------------------------ | -------------------------------------------------------------------- |
| `./gradlew build`        | Compile source + run tests + produce output JARs                     |
| `./gradlew run`          | Compile and run the application                                      |
| `./gradlew -q run`       | Run in quiet mode — suppresses Gradle's own log output               |
| `./gradlew -t -q run`    | Continuous mode — watches for file changes and re-runs automatically |
| `./gradlew test`         | Run tests only                                                       |
| `./gradlew clean`        | Delete all build artifacts (the `build/` directories)                |
| `./gradlew dependencies` | Print the full dependency tree for all configurations                |
| `./gradlew tasks`        | List all available tasks in the project                              |

The `-q` (quiet) flag is useful when you just want to see your program's output without Gradle's progress lines in the way.

The `-t` (continuous) flag is useful during development — save a file, and Gradle rebuilds and reruns immediately.

---

## Adding Dependencies

To add an external library, declare it in the `dependencies` block of `app/build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.squareup.moshi:moshi-kotlin:1.15.0")
    testImplementation("org.jetbrains.kotlin:kotlin-test")
}
```

The string format is `"group:artifact:version"`. You can find libraries and their coordinates on [Maven Central](https://search.maven.org/).

### `implementation` vs `testImplementation`

The scope controls when and where the dependency is available:

- **`implementation`** — available when compiling and running your main source code. This is the default choice for application dependencies.
- **`testImplementation`** — available only when compiling and running tests. The library is not included in the production output. Use this for testing frameworks like `kotlin-test` or `mockk`.

There's also `runtimeOnly` (available at runtime but not compile time) and `compileOnly` (available at compile time but not runtime), but `implementation` and `testImplementation` cover the vast majority of cases.

After adding a dependency, run `./gradlew build` and Gradle will download it from Maven Central automatically.

---

## How the Build Pipeline Works

When you run `./gradlew run`, Gradle goes through two phases:

### 1. Configuration phase

Gradle reads every `build.gradle.kts` and `settings.gradle.kts` file and constructs a **task graph** — a directed acyclic graph of all tasks and their dependencies. For example, `run` depends on `classes`, which depends on `compileKotlin`, which depends on having resolved your dependencies.

This phase is where configuration cache helps: if nothing in your build scripts changed since last time, Gradle can reload the task graph from cache and skip re-executing the scripts.

### 2. Execution phase

Gradle walks the task graph and runs each task that needs to run. Before executing a task, Gradle checks whether its inputs and outputs have changed since the last build. If nothing changed, the task is marked **UP-TO-DATE** and skipped.

For example, if you run `./gradlew build` twice in a row without changing anything, the second run will show almost everything as UP-TO-DATE and finish in a fraction of the time.

This is **incremental builds** — Gradle only does work that is actually necessary. It tracks task inputs (source files, dependencies, config values) and outputs (class files, JARs) as a fingerprint. If the fingerprint matches, the task is skipped.

The practical takeaway: the first build after a clean is slow. Every build after that is fast, unless you change something that invalidates a task's inputs.
