//! Four small ARC-style task families used by the upstream study script.
//!
//! These generators correspond to BDH-CQ paper section 6.2 / figure 5:
//! propagation, copying, ordering, and nesting.  They are pedagogical probes,
//! not the paper's undisclosed training mixture.  A task samples one persistent
//! layout/color scheme, renders easy demonstrations, then renders harder test
//! examples from the same rule.

use std::fmt;

use rand::{
    RngExt, SeedableRng,
    prelude::{SliceRandom, StdRng},
};

use crate::error::BdhError;

/// ARC's gray color, used as a copy anchor.
pub const GRAY: u8 = 5;

/// Non-black, non-gray colors available to the synthetic generators.
pub const COLORS: [u8; 8] = [1, 2, 3, 4, 6, 7, 8, 9];

/// A rectangular ten-color ARC grid stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    height: usize,
    width: usize,
    cells: Vec<u8>,
}

impl Grid {
    /// Construct a black (`0`) grid.
    pub fn zeros(height: usize, width: usize) -> Result<Self, BdhError> {
        if height == 0 || width == 0 {
            return Err(BdhError::InvalidGrid(
                "height and width must both be non-zero".into(),
            ));
        }
        Ok(Self {
            height,
            width,
            cells: vec![0; height * width],
        })
    }

    /// Construct and validate a grid from rows.
    pub fn from_rows(rows: Vec<Vec<u8>>) -> Result<Self, BdhError> {
        let height = rows.len();
        let width = rows.first().map(Vec::len).unwrap_or(0);
        if height == 0 || width == 0 {
            return Err(BdhError::InvalidGrid("a grid cannot be empty".into()));
        }
        if rows.iter().any(|row| row.len() != width) {
            return Err(BdhError::InvalidGrid("rows must be rectangular".into()));
        }
        if rows.iter().flatten().any(|color| *color > 9) {
            return Err(BdhError::InvalidGrid(
                "ARC colors must be integers in 0..=9".into(),
            ));
        }
        Ok(Self {
            height,
            width,
            cells: rows.into_iter().flatten().collect(),
        })
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Grids created by this type are never empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Borrow the row-major color buffer.
    pub fn cells(&self) -> &[u8] {
        &self.cells
    }

    /// Read one cell.
    pub fn get(&self, row: usize, column: usize) -> u8 {
        self.cells[row * self.width + column]
    }

    fn set(&mut self, row: usize, column: usize, color: u8) {
        self.cells[row * self.width + column] = color;
    }

    fn draw_motif(&mut self, row: usize, column: usize, motif: &Grid) {
        for motif_row in 0..motif.height {
            for motif_column in 0..motif.width {
                self.set(
                    row + motif_row,
                    column + motif_column,
                    motif.get(motif_row, motif_column),
                );
            }
        }
    }

    fn draw_border(&mut self, row: usize, column: usize, height: usize, width: usize, color: u8) {
        for offset in 0..width {
            self.set(row, column + offset, color);
            self.set(row + height - 1, column + offset, color);
        }
        for offset in 0..height {
            self.set(row + offset, column, color);
            self.set(row + offset, column + width - 1, color);
        }
    }

    /// Convert to owned rows, convenient for assertions and visualization.
    pub fn to_rows(&self) -> Vec<Vec<u8>> {
        self.cells.chunks(self.width).map(<[u8]>::to_vec).collect()
    }
}

impl fmt::Display for Grid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.height {
            if row > 0 {
                writeln!(formatter)?;
            }
            for column in 0..self.width {
                let color = self.get(row, column);
                if color == 0 {
                    write!(formatter, ".")?;
                } else {
                    write!(formatter, "{color}")?;
                }
            }
        }
        Ok(())
    }
}

/// One demonstration or held-out query/answer pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    /// Difficulty parameter whose meaning depends on the family.
    pub level: usize,
    /// Input grid shown to the model.
    pub input: Grid,
    /// Rule-transformed target grid.
    pub output: Grid,
}

/// Persistent values sampled once and shared by every example in a task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskParameters {
    /// Canvas and seed-bar placement for propagation.
    Propagation {
        /// Canvas height.
        height: usize,
        /// Canvas width.
        width: usize,
        /// Vertical bar height.
        bar_height: usize,
        /// First bar row.
        first_row: usize,
        /// Bar color.
        color: u8,
    },
    /// Source motif and block layout for copying.
    Copy {
        /// Side length of one motif/block.
        block_size: usize,
        /// Number of block rows.
        block_rows: usize,
        /// Number of block columns.
        block_columns: usize,
        /// Colored motif copied to gray anchors.
        motif: Grid,
        /// Source block coordinate from which the rule is inferred.
        source: (usize, usize),
    },
    /// Canvas dimensions for height ordering.
    Order {
        /// Canvas height.
        height: usize,
        /// Canvas width.
        width: usize,
    },
    /// Center region and frame palette for nesting.
    Nesting {
        /// Center-region width.
        region_width: usize,
        /// Center-region height.
        region_height: usize,
        /// Color written inside the innermost frame.
        region_color: u8,
        /// Frame colors from inner to outer depth.
        frame_colors: Vec<u8>,
        /// Maximum allocated frame depth.
        max_depth: usize,
    },
}

/// Complete in-context task: demonstrations plus held-out examples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskData {
    /// Family name used by the upstream registry.
    pub name: &'static str,
    /// Layout shared across train and test examples.
    pub parameters: TaskParameters,
    /// Easy examples presented as context.
    pub train: Vec<Example>,
    /// Usually harder examples used as queries.
    pub test: Vec<Example>,
}

/// One of the four paper-inspired synthetic families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskFamily {
    /// Extend a colored vertical seed bar to the right boundary.
    Propagation {
        /// Enables upstream's smaller demonstration regime when present.
        size: Option<usize>,
    },
    /// Copy a colored motif to a chosen number of gray anchors.
    Copy {
        /// Enables upstream's smaller demonstration regime when present.
        size: Option<usize>,
    },
    /// Reorder colored bars from shortest to tallest.
    Order {
        /// Maximum number of bars in the small regime.
        size: Option<usize>,
    },
    /// Recolor the innermost region of nested frames.
    Nesting {
        /// Maximum nesting depth in the small regime.
        size: Option<usize>,
    },
}

impl TaskFamily {
    /// All four families at their full default scale.
    pub fn all() -> [Self; 4] {
        [
            Self::Propagation { size: None },
            Self::Copy { size: None },
            Self::Order { size: None },
            Self::Nesting { size: None },
        ]
    }

    /// Return the same family with the upstream small-demo `size` override.
    pub fn with_size(self, size: usize) -> Self {
        match self {
            Self::Propagation { .. } => Self::Propagation { size: Some(size) },
            Self::Copy { .. } => Self::Copy { size: Some(size) },
            Self::Order { .. } => Self::Order { size: Some(size) },
            Self::Nesting { .. } => Self::Nesting { size: Some(size) },
        }
    }

    /// Stable upstream registry name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Propagation { .. } => "propagation",
            Self::Copy { .. } => "copy",
            Self::Order { .. } => "order",
            Self::Nesting { .. } => "nesting",
        }
    }

    /// Generate three easy demonstrations and two held-out examples.
    pub fn generate(self, seed: u64) -> Result<TaskData, BdhError> {
        self.generate_with(seed, 3, 2, None)
    }

    /// Generate a task with explicit counts and optional held-out level range.
    pub fn generate_with(
        self,
        seed: u64,
        num_demonstrations: usize,
        num_tests: usize,
        test_levels: Option<(usize, usize)>,
    ) -> Result<TaskData, BdhError> {
        if num_demonstrations == 0 || num_tests == 0 {
            return Err(BdhError::InvalidGrid(
                "a task needs at least one demonstration and one test".into(),
            ));
        }
        self.validate_size()?;

        let mut rng = StdRng::seed_from_u64(seed);
        let parameters = self.sample(&mut rng)?;
        let (demo_levels, default_test_levels) = self.level_ranges();
        let test_levels = test_levels.unwrap_or(default_test_levels);
        if test_levels.0 > test_levels.1 {
            return Err(BdhError::InvalidGrid("test level range is reversed".into()));
        }

        let train = (0..num_demonstrations)
            .map(|_| {
                let level = inclusive(&mut rng, demo_levels.0, demo_levels.1);
                self.render(&mut rng, level, &parameters)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let test = (0..num_tests)
            .map(|_| {
                let level = inclusive(&mut rng, test_levels.0, test_levels.1);
                self.render(&mut rng, level, &parameters)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(TaskData {
            name: self.name(),
            parameters,
            train,
            test,
        })
    }

    /// Generate a task whose held-out examples have exactly `level`.
    pub fn at_level(
        self,
        seed: u64,
        level: usize,
        num_demonstrations: usize,
        num_tests: usize,
    ) -> Result<TaskData, BdhError> {
        self.generate_with(seed, num_demonstrations, num_tests, Some((level, level)))
    }

    fn size(self) -> Option<usize> {
        match self {
            Self::Propagation { size }
            | Self::Copy { size }
            | Self::Order { size }
            | Self::Nesting { size } => size,
        }
    }

    fn validate_size(self) -> Result<(), BdhError> {
        if let Some(size) = self.size() {
            let valid = match self {
                Self::Propagation { .. } => size >= 3,
                Self::Copy { .. } => size >= 2,
                // `size` controls available columns, while rendered examples
                // still contain at most four bars in the small regime. Values
                // above eight are therefore valid in the Python generator.
                Self::Order { .. } => size >= 4,
                Self::Nesting { .. } => (2..=7).contains(&size),
            };
            if !valid {
                return Err(BdhError::InvalidGrid(format!(
                    "size {size} is not valid for {}",
                    self.name()
                )));
            }
        }
        Ok(())
    }

    fn level_ranges(self) -> ((usize, usize), (usize, usize)) {
        match (self, self.size().is_some()) {
            (Self::Propagation { .. }, false) => ((1, 3), (2, 8)),
            (Self::Propagation { .. }, true) => ((1, 2), (2, 4)),
            (Self::Copy { .. }, false) => ((1, 2), (2, 4)),
            (Self::Copy { .. }, true) => ((1, 2), (2, 3)),
            (Self::Order { .. }, false) => ((2, 4), (5, 8)),
            (Self::Order { .. }, true) => ((2, 3), (3, 4)),
            (Self::Nesting { .. }, false) => ((1, 3), (4, 5)),
            (Self::Nesting { .. }, true) => ((1, 2), (2, 3)),
        }
    }

    fn sample(self, rng: &mut StdRng) -> Result<TaskParameters, BdhError> {
        match self {
            Self::Propagation { size } => {
                let (height, width) = match size {
                    Some(size) => (size, size),
                    None => (inclusive(rng, 6, 10), inclusive(rng, 10, 12)),
                };
                let bar_height = inclusive(rng, 2, 4.min(height - 1));
                Ok(TaskParameters::Propagation {
                    height,
                    width,
                    bar_height,
                    first_row: inclusive(rng, 0, height - bar_height),
                    color: choose_color(rng),
                })
            }
            Self::Copy { size } => {
                let (block_size, block_rows, block_columns) = if size.is_some() {
                    (2, 3, 3)
                } else {
                    (
                        if rng.random::<bool>() { 2 } else { 3 },
                        inclusive(rng, 3, 4),
                        inclusive(rng, 3, 4),
                    )
                };
                let mut motif = Grid::zeros(block_size, block_size)?;
                let count = inclusive(rng, 2, (block_size * block_size).min(COLORS.len()));
                let mut cell_indices: Vec<_> = (0..block_size * block_size).collect();
                cell_indices.shuffle(rng);
                let mut colors = COLORS;
                colors.shuffle(rng);
                for (cell, color) in cell_indices.into_iter().zip(colors).take(count) {
                    motif.cells[cell] = color;
                }
                Ok(TaskParameters::Copy {
                    block_size,
                    block_rows,
                    block_columns,
                    motif,
                    source: (
                        rng.random_range(0..block_rows),
                        rng.random_range(0..block_columns),
                    ),
                })
            }
            Self::Order { size } => {
                let max_bars = size.unwrap_or(8);
                Ok(TaskParameters::Order {
                    height: if size.is_some() {
                        inclusive(rng, 8, 9)
                    } else {
                        inclusive(rng, 10, 12)
                    },
                    width: 2 * max_bars + 2,
                })
            }
            Self::Nesting { size } => {
                let max_depth = size.unwrap_or(5);
                let mut colors = COLORS;
                colors.shuffle(rng);
                Ok(TaskParameters::Nesting {
                    region_width: inclusive(rng, 1, 3),
                    region_height: inclusive(rng, 1, 3),
                    region_color: colors[0],
                    frame_colors: colors[1..max_depth + 1].to_vec(),
                    max_depth,
                })
            }
        }
    }

    fn render(
        self,
        rng: &mut StdRng,
        level: usize,
        parameters: &TaskParameters,
    ) -> Result<Example, BdhError> {
        let (input, output) = match parameters {
            TaskParameters::Propagation {
                height,
                width,
                bar_height,
                first_row,
                color,
            } => {
                let raw_column = *width as isize - 1 - level as isize;
                // NumPy accepts negative indices.  Upstream's deliberately
                // tiny `size=3` test regime can produce gaps 3 or 4, so keep
                // that wraparound behavior instead of rejecting them.
                if raw_column < -(*width as isize) {
                    return Err(BdhError::InvalidGrid(
                        "propagation gap lies outside the canvas".into(),
                    ));
                }
                let column = if raw_column < 0 {
                    (*width as isize + raw_column) as usize
                } else {
                    raw_column as usize
                };
                let mut input = Grid::zeros(*height, *width)?;
                for row in *first_row..first_row + bar_height {
                    input.set(row, column, *color);
                }
                let mut output = input.clone();
                for row in *first_row..first_row + bar_height {
                    for column in column..*width {
                        output.set(row, column, *color);
                    }
                }
                (input, output)
            }
            TaskParameters::Copy {
                block_size,
                block_rows,
                block_columns,
                motif,
                source,
            } => {
                let mut spots: Vec<_> = (0..*block_rows)
                    .flat_map(|row| (0..*block_columns).map(move |column| (row, column)))
                    .filter(|spot| spot != source)
                    .collect();
                spots.shuffle(rng);
                if level > spots.len() {
                    return Err(BdhError::InvalidGrid(
                        "copy level exceeds the number of anchor locations".into(),
                    ));
                }
                let anchors = &spots[..level];
                let mut input = Grid::zeros(block_size * block_rows, block_size * block_columns)?;
                for &(row, column) in anchors {
                    input.set(row * block_size, column * block_size, GRAY);
                }
                // This intentionally preserves upstream's exact source placement:
                // source is not multiplied by block_size, whereas anchors are.
                input.draw_motif(source.0, source.1, motif);
                let mut output = input.clone();
                for &(row, column) in anchors {
                    output.draw_motif(row * block_size, column * block_size, motif);
                }
                (input, output)
            }
            TaskParameters::Order { height, width } => {
                if level > COLORS.len() || level > (width - 1) / 2 {
                    return Err(BdhError::InvalidGrid(
                        "order level exceeds available colors or columns".into(),
                    ));
                }
                let mut heights: Vec<_> = (1..=level).collect();
                heights.shuffle(rng);
                let mut colors = COLORS;
                colors.shuffle(rng);
                let colors = &colors[..level];
                let mut columns: Vec<_> = (1..width - 1).step_by(2).collect();
                columns.shuffle(rng);

                let mut input = Grid::zeros(*height, *width)?;
                for ((bar_height, color), column) in heights.iter().zip(colors).zip(columns) {
                    for row in height - bar_height..*height {
                        input.set(row, column, *color);
                    }
                }

                let mut order: Vec<_> = (0..level).collect();
                order.sort_by_key(|index| heights[*index]);
                let mut output = Grid::zeros(*height, *width)?;
                for (column, index) in order.into_iter().enumerate() {
                    for row in height - heights[index]..*height {
                        output.set(row, column, colors[index]);
                    }
                }
                (input, output)
            }
            TaskParameters::Nesting {
                region_width,
                region_height,
                region_color,
                frame_colors,
                max_depth,
            } => {
                if level > *max_depth {
                    return Err(BdhError::InvalidGrid(
                        "nesting level exceeds allocated frame colors".into(),
                    ));
                }
                let padding = 2 * max_depth;
                let mut input =
                    Grid::zeros(region_height + 2 * padding, region_width + 2 * padding)?;
                for depth in 1..=level {
                    input.draw_border(
                        padding - 2 * depth,
                        padding - 2 * depth,
                        region_height + 4 * depth,
                        region_width + 4 * depth,
                        frame_colors[depth - 1],
                    );
                }
                let mut output = input.clone();
                for row in padding..padding + region_height {
                    for column in padding..padding + region_width {
                        output.set(row, column, *region_color);
                    }
                }
                (input, output)
            }
        };

        Ok(Example {
            level,
            input,
            output,
        })
    }
}

fn inclusive(rng: &mut StdRng, low: usize, high: usize) -> usize {
    rng.random_range(low..=high)
}

fn choose_color(rng: &mut StdRng) -> u8 {
    COLORS[rng.random_range(0..COLORS.len())]
}
