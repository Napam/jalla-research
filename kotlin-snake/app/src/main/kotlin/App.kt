import java.awt.*
import java.awt.event.*
import javax.swing.*
import java.util.LinkedList

// ── Constants ────────────────────────────────────────────────────────────────
const val CELL = 20          // pixel size of one grid cell
const val COLS = 30          // grid columns  → window width  = 600
const val ROWS = 30          // grid rows     → window height = 600
const val TICK = 120         // ms per game tick (~8 FPS feel, smooth for Snake)

// ── Data ─────────────────────────────────────────────────────────────────────
data class Cell(val x: Int, val y: Int)

enum class Dir { UP, DOWN, LEFT, RIGHT }

// ── Game state ────────────────────────────────────────────────────────────────
class SnakeGame {
    var snake = LinkedList<Cell>()
    var food = Cell(0, 0)
    var dir = Dir.RIGHT
    var nextDir = Dir.RIGHT   // buffered input — applied at next tick
    var score = 0
    var over = false

    init {
        reset()
    }

    fun reset() {
        snake.clear()
        // Start with a 3-cell snake in the middle
        val cx = COLS / 2
        val cy = ROWS / 2
        snake.addFirst(Cell(cx, cy))
        snake.addFirst(Cell(cx + 1, cy))
        snake.addFirst(Cell(cx + 2, cy))
        dir = Dir.RIGHT
        nextDir = Dir.RIGHT
        score = 0
        over = false
        spawnFood()
    }

    fun spawnFood() {
        val occupied = snake.toSet()
        val free = (0 until COLS).flatMap { x -> (0 until ROWS).map { y -> Cell(x, y) } }
            .filter { it !in occupied }
        food = if (free.isNotEmpty()) free.random() else Cell(0, 0)
    }

    // Called once per timer tick
    fun update() {
        if (over) return

        // Commit buffered direction (ignore 180° reversal)
        val forbidden = when (dir) {
            Dir.UP -> Dir.DOWN; Dir.DOWN -> Dir.UP
            Dir.LEFT -> Dir.RIGHT; Dir.RIGHT -> Dir.LEFT
        }
        if (nextDir != forbidden) dir = nextDir

        val head = snake.first()
        val next = when (dir) {
            Dir.UP -> Cell(head.x, head.y - 1)
            Dir.DOWN -> Cell(head.x, head.y + 1)
            Dir.LEFT -> Cell(head.x - 1, head.y)
            Dir.RIGHT -> Cell(head.x + 1, head.y)
        }

        // Wall collision
        if (next.x !in 0 until COLS || next.y !in 0 until ROWS) {
            over = true; return
        }
        // Self collision
        if (next in snake) {
            over = true; return
        }

        snake.addFirst(next)
        if (next == food) {
            score++
            spawnFood()
        } else {
            snake.removeLast()   // no growth — keep length
        }
    }
}

// ── Panel (render + input) ────────────────────────────────────────────────────
class GamePanel : JPanel(), KeyListener {
    private val game = SnakeGame()
    private val timer = Timer(TICK) { game.update(); repaint() }

    // Colour palette
    private val colBg = Color(15, 15, 20)
    private val colGrid = Color(30, 30, 38)
    private val colSnakeH = Color(100, 220, 100)
    private val colSnakeB = Color(60, 160, 60)
    private val colFood = Color(230, 80, 80)
    private val colOverlay = Color(0, 0, 0, 160)
    private val colText = Color(220, 220, 220)
    private val colScore = Color(100, 220, 100)

    private val fontHud = Font("Monospaced", Font.BOLD, 16)
    private val fontBig = Font("Monospaced", Font.BOLD, 36)
    private val fontSmall = Font("Monospaced", Font.PLAIN, 14)

    init {
        preferredSize = Dimension(COLS * CELL, ROWS * CELL)
        background = colBg
        isFocusable = true
        addKeyListener(this)
        timer.start()
    }

    // ── Render ────────────────────────────────────────────────────────────────
    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)

        drawGrid(g2)
        drawFood(g2)
        drawSnake(g2)
        drawHud(g2)
        if (game.over) drawGameOver(g2)
    }

    private fun drawGrid(g: Graphics2D) {
        g.color = colGrid
        for (x in 0..COLS) g.drawLine(x * CELL, 0, x * CELL, ROWS * CELL)
        for (y in 0..ROWS) g.drawLine(0, y * CELL, COLS * CELL, y * CELL)
    }

    private fun drawFood(g: Graphics2D) {
        val pad = 3
        g.color = colFood
        g.fillOval(
            game.food.x * CELL + pad, game.food.y * CELL + pad,
            CELL - pad * 2, CELL - pad * 2
        )
        // Shine dot
        g.color = Color(255, 180, 180)
        g.fillOval(game.food.x * CELL + pad + 3, game.food.y * CELL + pad + 2, 4, 4)
    }

    private fun drawSnake(g: Graphics2D) {
        game.snake.forEachIndexed { i, cell ->
            g.color = if (i == 0) colSnakeH else colSnakeB
            val pad = if (i == 0) 1 else 2
            g.fillRoundRect(
                cell.x * CELL + pad, cell.y * CELL + pad,
                CELL - pad * 2, CELL - pad * 2,
                6, 6
            )
            // Eyes on head
            if (i == 0) {
                g.color = colBg
                val (ex1, ey1, ex2, ey2) = when (game.dir) {
                    Dir.RIGHT -> intArrayOf(
                        cell.x * CELL + 13, cell.y * CELL + 5,
                        cell.x * CELL + 13, cell.y * CELL + 13
                    )

                    Dir.LEFT -> intArrayOf(
                        cell.x * CELL + 4, cell.y * CELL + 5,
                        cell.x * CELL + 4, cell.y * CELL + 13
                    )

                    Dir.UP -> intArrayOf(
                        cell.x * CELL + 5, cell.y * CELL + 4,
                        cell.x * CELL + 13, cell.y * CELL + 4
                    )

                    Dir.DOWN -> intArrayOf(
                        cell.x * CELL + 5, cell.y * CELL + 13,
                        cell.x * CELL + 13, cell.y * CELL + 13
                    )
                }
                g.fillOval(ex1, ey1, 4, 4)
                g.fillOval(ex2, ey2, 4, 4)
            }
        }
    }

    private fun drawHud(g: Graphics2D) {
        g.font = fontHud
        g.color = colScore
        g.drawString("Score: ${game.score}", 8, 20)
        val len = "Length: ${game.snake.size}"
        g.color = colText
        g.drawString(len, width - g.fontMetrics.stringWidth(len) - 8, 20)
    }

    private fun drawGameOver(g: Graphics2D) {
        // Dim overlay
        g.color = colOverlay
        g.fillRect(0, 0, width, height)

        val cx = width / 2

        g.font = fontBig
        g.color = colFood
        val over = "GAME OVER"
        g.drawString(over, cx - g.fontMetrics.stringWidth(over) / 2, height / 2 - 30)

        g.font = fontHud
        g.color = colScore
        val sc = "Score: ${game.score}   Length: ${game.snake.size}"
        g.drawString(sc, cx - g.fontMetrics.stringWidth(sc) / 2, height / 2 + 10)

        g.font = fontSmall
        g.color = colText
        val hint = "Press R or Space to restart"
        g.drawString(hint, cx - g.fontMetrics.stringWidth(hint) / 2, height / 2 + 42)
    }

    // ── Input ─────────────────────────────────────────────────────────────────
    override fun keyPressed(e: KeyEvent) {
        when (e.keyCode) {
            KeyEvent.VK_UP, KeyEvent.VK_W -> game.nextDir = Dir.UP
            KeyEvent.VK_DOWN, KeyEvent.VK_S -> game.nextDir = Dir.DOWN
            KeyEvent.VK_LEFT, KeyEvent.VK_A -> game.nextDir = Dir.LEFT
            KeyEvent.VK_RIGHT, KeyEvent.VK_D -> game.nextDir = Dir.RIGHT
            KeyEvent.VK_R, KeyEvent.VK_SPACE -> if (game.over) game.reset()
        }
    }

    override fun keyReleased(e: KeyEvent) {}
    override fun keyTyped(e: KeyEvent) {}
}

// ── Entry point ───────────────────────────────────────────────────────────────
fun main() {
    SwingUtilities.invokeLater {
        val frame = JFrame("Snake").apply {
            defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            isResizable = false
            add(GamePanel())
            pack()
            setLocationRelativeTo(null)   // center on screen
            isVisible = true
        }
        // Make sure the panel gets keyboard focus immediately
        frame.contentPane.getComponent(0).requestFocusInWindow()
    }
}
