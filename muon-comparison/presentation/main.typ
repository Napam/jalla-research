#import "@preview/lovelace:0.3.1": pseudocode-list
#set page(
  paper: "presentation-16-9",
  fill: rgb(14.5%, 15.3%, 19.2%),
  numbering: "1",
)

#let algo-stroke = 1.5pt + rgb(90%, 90%, 90%)

#set text(
  font: "New Computer Modern",
  fill: rgb(90%, 90%, 90%),
  size: 16pt,
)

#set math.mat(delim: "[")

#let sgn = math.op("sgn")

#let spaced-figure(caption: none, body) = {
  v(0.5em)
  figure(caption: caption, body)
}

#let accent-color = rgb(94.5%, 69.4%, 34.5%)

#show heading: content => {
  set text(fill: accent-color)
  content
  v(0.5em)
}

//========================================================= f

#box(width: 100%, height: 100%, stroke: accent-color)[
  #set align(center)
  #v(1em)
  #text(size: 14pt, fill: accent-color)[
    Naphat Amundsen presents:
  ]

  #place(center + horizon)[
    #box(inset: (bottom: 16pt), stroke: (bottom: accent-color))[
      #title[
        #text(fill: accent-color, size: 42pt)[Muon]
      ]

      #text(weight: "bold")[M]oment#text(weight: "bold")[U]m #text(
        weight: "bold",
      )[O]rthogonalized by #text(weight: "bold")[N]ewton-Schulz
    ]
  ]

  #align(bottom)[
    #box(inset: (bottom: 1em))[
      #text(size: 14pt)[
        #datetime.today().display()
      ]
    ]
  ]

]

#pagebreak()

= Intro
- AdamW and similar optimizers essentially do momentum on each weight
  individually.
  - Empirically shown that the momentum matrix to have a tendency to become
    low-rank.
- Muon takes a more wholistic view of the momentum values together


#page[
  #block(width: 100%)[
    #set align(center)
    === Standard AdamW

    #pseudocode-list(
      booktabs: true,
      booktabs-stroke: algo-stroke,
      title: [
        *Require :* $gamma$ (lr), $beta_1, beta_2$ (betas), $theta_0$ (params),
        $f(theta)$ (objective), $epsilon$ (epsilon), \
        #h(2.65em) $lambda$ (weight decay)
      ],
    )[
      + Initialize: $m_0 <- 0$ (first moment), $v_0 <- 0$ (second moment),
        $v_0^max <- 0$
      + *for* $t = 1$ to $dots$ *do*
        + Compute gradient $g_t <- nabla_theta f_t (theta_(t-1))$
        + Weight decay $theta_t <- theta_(t-1) - gamma lambda theta_(t-1)$
        + Update momentum $m_t <- beta_1 m_(t-1) + (1 - beta_1) g_t$
        + Update RMS $v_t <- beta_2 v_(t-1) + (1 - beta_2) g_t^2$
        + Bias correct momentum $hat(m)_t <- m_t \/ (1 - beta_1^t)$
        + Bias correct RMS $hat(v)_t <- v_t \/ (1 - beta_2^t)$
        + Update parameters
          $theta_t <- theta_t - gamma hat(m)_t \/ (sqrt(hat(v)_t) + epsilon)$
      + *return* $theta_t$
    ]
  ]
]

= Understanding the issue

AdamW (and other momentum-based) optimizers only look at each weights
individually.

- Let's look closer at momentum update in the case of a $(N , M)$ matrix:
$
  m arrow.l beta m + frac(partial L, partial theta),
  quad
  m = mat(
    m_11, m_12, m_13;
    m_21, m_22, m_23;
    m_31, m_32, m_33;
    m_41, m_42, m_43;
  )
$

what is observed is that the momentum matrix during training have tendencies to
become "practically low-rank": e.g.
$
  mat(
    1, 1, 1;
    1, 0.99, 1.001;
    1, 1.01, 1.002;
    1, 1.01, 1.003;
  )
$

If we think of gradient descent as a ball rolling down the loss landscape, the
problem is that the ball tends build momentum and "shoot" in a certain
direction.

#spaced-figure(
  caption: "Visualization of gradient steps over a loss landscape",
)[
  #image("loss-landscape.png", height: 60%)
]

#figure(caption: "Visualization of gradient steps in a \"bowl\"")[
  #image("optimization-algorithms-race.png", height: 90%)
]




So, how can one condition the momentum matrix to be more well-behaved?

- The regular "element-by-element" optimizers cannot solve this issue directly,
  because they have to look at the momentums more wholistically, which they
  don't.

- Thus have a method that "looks at the whole matrix".

- Muon: let's orthogonalize the momentum matrix!


#pagebreak()

== Orthogonalization recap

$"Ortho"(bold(M)) = "argmin"_bold(O) {||bold(O)-bold(M)||_F} "such that" O O^T = I "or" O^T O = I$

#text(size: 12pt)[
  Rant: the naming is stupid, the name is called "orthogonal", but the
  mathematical definition here implies orthonormal.
]

- In English: find an orthonormal matrix that is as similar to $bold(M)$ as
  possible

- A theoretically sound solution is to use singular value decomposition (SVD):
  - Factorize the matrix $M$ to the form $M = U Sigma V^T$
  - Where $U$ is a unitary matrix: $U^T U = U U^T = I$
  - One can then do $U V^T$ to get an orthonormal version of $M$

#pagebreak()

Problem solved, right?
#v(1em)
#box(width: 100%)[
  #set align(center)
  #image("right-meme.png", width: 80%)
]


#pagebreak()

#image("reality-slap.jpg")

Problem: Calculating SVD for every single backprop for every single matrix in a
model is too resource intensive.

They do a trick: use Newton-Schulz iteration to approximate the
orthogonalization, which is much more efficient and can be easily accelerated by
GPUs.

#page()[
  #set text(size: 13pt)

  == Newton-Schulz 101
  The Newton-Schulz iteration is a mathematical approximation to approximate the
  Matrix Sign Function.

  #v(0.5em)

  *For Scalars ($x$):* The sign function $sgn(x)$ strips away the magnitude and
  keeps only the direction (returning $-1$, $0$, or $+1$).

  #v(0.5em)

  *For Matrices ($M$):* The sign function $sgn(M)$ strips away the stretching
  factors (singular values) and keeps only the orientation.

  #v(0.5em)

  *The SVD Connection:*
  - Given the decomposition $M = U Sigma V^T$, the "sign" of $M$ is defined as
    $U V^T$.
  - If we force $Sigma$ to become the Identity Matrix $I$, we essentially get
    the "sign" of $M$.

  #v(0.5em)

  *Newton-Schulz iteration:*
  Newton-Schulz iteration is a method to approximate the matrix sign function
  without explicitly computing the SVD.
  - It uses an *odd polynomial* (e.g., $1.5X - 0.5X^3$) to iteratively "squash"
    the singular values in $Sigma$ toward $1$.
  - Because the polynomial is *odd*, it preserves the direction of the gradient
    while only using matrix multiplications.

  #v(0.5em)

  *The Geometric Result:*
  $U V^T$ is the closest orthonormal matrix to the original gradient. It turns a
  "squashed ellipse" (an ill-conditioned gradient) into a "perfect sphere" (an
  orthonormal update).
]

#block(width: 100%)[
  #set align(center)

  = The Muon Optimizer

  #pseudocode-list(
    booktabs: true,
    booktabs-stroke: algo-stroke,
    title: [
      *Require:* Learning rate $eta$, momentum $mu$
    ],
    line-numbering: "1:",
  )[
    + Initialize $B_0 <- 0$
    + *for* $t = 1, dots$ *do*
      + Compute gradient $G_t <- nabla_theta cal(L)_t (theta_(t-1))$
      + $B_t <- mu B_(t-1) + G_t$
      + $O_t <-$ NewtonSchulz5$(B_t)$
      + Update parameters $theta_t <- theta_(t-1) - eta O_t$
    + *end for*
    + *return* $theta_t$
  ]
  Note: there are some differences between the pseudocode and the actual
  implementation
]

