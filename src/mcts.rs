use ego_tree::{NodeId, Tree};

pub trait MctsGame: Clone {
    type Action: Copy;
    type Player: Copy + PartialEq;

    fn legal_actions(&self) -> Vec<Self::Action>;
    fn play(&mut self, action: Self::Action);
    fn player(&self) -> Self::Player;
    fn state(&self, player: Self::Player) -> MctsState;
}

#[derive(Debug, PartialEq)]
pub enum MctsState {
    Win,
    Lose,
    Draw,
    Unfinished,
}

pub struct Mcts<Game: MctsGame> {
    tree: Tree<MctsNode<Game>>,
    player: Game::Player,
}

impl<Game: MctsGame + std::fmt::Debug> Mcts<Game> {
    const EXPLORATION: f32 = core::f32::consts::SQRT_2;

    pub fn new(game: Game) -> Self {
        Self {
            player: game.player(),
            tree: Tree::new(MctsNode::new(game, None)),
        }
    }

    pub fn search(&mut self, iterations: u64) {
        for _ in 0..iterations {
            let leaf = self.select_leaf();
            let new_leaf = self.expand(leaf).unwrap_or(leaf);
            let state = self.simulate(new_leaf);
            self.back_propagate(new_leaf, state);
        }
        // println!("{:?}", self.tree);
    }

    pub fn best_action(&self) -> (Game::Action, f32) {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_child = None;

        let mut child = self.tree.root().first_child().map(|x| x.id());
        while let Some(next_child) = child {
            let score = unsafe { self.tree.get_unchecked(next_child) }
                .value()
                .score(self.player);
            if score > best_score {
                best_score = score;
                best_child = child;
            }
            child = unsafe { self.tree.get_unchecked(next_child) }
                .next_sibling()
                .map(|x| x.id());
        }
        let node = unsafe { self.tree.get_unchecked(best_child.unwrap()) }.value();
        (node.last_action().unwrap(), node.score(self.player))
    }

    fn win(&mut self, node_id: NodeId) {
        unsafe { self.tree.get_unchecked_mut(node_id) }.value().won += 1;
    }

    fn lose(&mut self, node_id: NodeId) {
        unsafe { self.tree.get_unchecked_mut(node_id) }.value().lost += 1;
    }

    fn draw(&mut self, node_id: NodeId) {
        unsafe { self.tree.get_unchecked_mut(node_id) }
            .value()
            .drawn += 1;
    }

    fn selection_score(&self, node: NodeId) -> f32 {
        let node = unsafe { self.tree.get_unchecked(node) };
        let parent = node.parent().unwrap().value();
        let node = node.value();
        node.score(self.player)
            + Self::EXPLORATION * f32::sqrt(f32::ln(parent.played() as f32) / node.played() as f32)
    }

    fn select_leaf(&mut self) -> NodeId {
        let mut node_id = self.tree.root().id();
        while !self.is_leaf(node_id) {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_child_id = None;
            let mut child_id = unsafe { self.tree.get_unchecked_mut(node_id) }
                .first_child()
                .map(|x| x.id());
            while child_id.is_some() {
                let score = self.selection_score(child_id.unwrap());
                if score > best_score {
                    best_score = score;
                    best_child_id = child_id;
                }
                child_id = unsafe { self.tree.get_unchecked_mut(child_id.unwrap()) }
                    .next_sibling()
                    .map(|x| x.id());
            }
            node_id = self.tree.get_mut(best_child_id.unwrap()).unwrap().id();
        }
        node_id
    }

    fn expand(&mut self, leaf: NodeId) -> Option<NodeId> {
        let mut node = unsafe { self.tree.get_unchecked_mut(leaf) };
        let mut game = node.value().game().clone();
        let actions = node.value().actions_mut();
        if actions.is_empty() {
            return None;
        }
        let len = actions.len();
        let index = if len == 1 {
            0
        } else {
            rand::random::<usize>() % len
        };
        let action = actions.swap_remove(index);
        game.play(action);
        Some(
            unsafe { self.tree.get_unchecked_mut(leaf) }
                .append(MctsNode::new(game, Some(action)))
                .id(),
        )
    }

    fn back_propagate(&mut self, leaf: NodeId, state: MctsState) {
        let f = match state {
            MctsState::Unfinished => unreachable!(),
            MctsState::Win => Self::win,
            MctsState::Lose => Self::lose,
            MctsState::Draw => Self::draw,
        };
        let mut node = Some(leaf);
        while let Some(next_node) = node {
            f(self, next_node);
            node = unsafe { self.tree.get_unchecked(next_node) }
                .parent()
                .map(|x| x.id());
        }
    }

    fn simulate(&mut self, leaf: NodeId) -> MctsState {
        let mut game = unsafe { self.tree.get_unchecked(leaf) }
            .value()
            .game()
            .clone();
        loop {
            let state = game.state(self.player);
            match state {
                MctsState::Unfinished => {
                    let actions = game.legal_actions();
                    let len = actions.len();
                    let index = if len == 1 {
                        0
                    } else {
                        rand::random::<usize>() % len
                    };
                    let action = actions[index];
                    game.play(action);
                }
                state => {
                    return state;
                }
            }
        }
    }

    fn is_leaf(&self, node_id: NodeId) -> bool {
        let node = unsafe { self.tree.get_unchecked(node_id) };
        if let MctsState::Unfinished = node.value().game().state(node.value().game().player()) {
            !node.has_children() || !node.value().actions().is_empty()
        } else {
            true
        }
    }
}

#[derive(Debug)]
struct MctsNode<Game: MctsGame> {
    game: Game,
    last_action: Option<Game::Action>,
    actions: Vec<Game::Action>,
    won: u32,
    lost: u32,
    drawn: u32,
}

impl<Game: MctsGame> MctsNode<Game> {
    fn new(game: Game, last_action: Option<Game::Action>) -> Self {
        Self {
            actions: game.legal_actions(),
            game,
            last_action,
            won: 0,
            lost: 0,
            drawn: 0,
        }
    }

    fn game(&self) -> &Game {
        &self.game
    }

    fn actions(&self) -> &[Game::Action] {
        &self.actions
    }

    fn actions_mut(&mut self) -> &mut Vec<Game::Action> {
        &mut self.actions
    }

    fn last_action(&self) -> Option<Game::Action> {
        self.last_action
    }

    fn score(&self, player: Game::Player) -> f32 {
        let score = (2 * self.won + self.drawn) as f32 / (2 * self.played()) as f32;
        if player == self.game().player() {
            score
        } else {
            1f32 - score
        }
    }

    fn played(&self) -> u32 {
        self.won + self.lost + self.drawn
    }
}
