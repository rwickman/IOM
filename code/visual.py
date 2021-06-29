import pygame, sys

from simulator import InventoryProduct, InventoryNode, DemandNode
from policy import PolicyResults 

pygame.init()

COLUMN_SEP = "    "
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255,0,0)

# The y offset below at which node info will display
NODE_INFO_OFFSET = 32

CIRCLE_RADIUS = 20
LINE_WIDTH = 5

class Visual:
    def __init__(self, args, inv_nodes: list[InventoryNode]):
        self.args = args
        self._inv_nodes = inv_nodes

        # Set width and height of the screen
        self.width, self.height = self.args.screen_size, self.args.screen_size

        # Simulation grid size
        self.sim_grid_size = [self.args.coord_bounds, self.args.coord_bounds]

        # Visualization Screen Size to Simulator Grid Size ratio
        # Includes paddding for the actual area that will define the simulator grid 
        self._pos_ratio = (
            (self.width - self.args.screen_padding) / self.sim_grid_size[0],
            (self.height - self.args.screen_padding) / self.sim_grid_size[1],
        )

        # Create the screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.font = pygame.font.Font('freesansbold.ttf', self.args.font_size)
        self._total_reward = 0
    
    def _sim_to_screen(self,
                    pos: list[float, float]) -> list[float, float]:    
        """Transform simulator position to screen position."""          
        for i in range(2):
            # Scale to [0,1] range
            pos[i] = (pos[i] - -self.args.coord_bounds) / \
                (self.args.coord_bounds - -self.args.coord_bounds)

            # Scale to padded screen range [padding, screen_size - padding]
            a = self.args.screen_padding
            b = self.args.screen_size - self.args.screen_padding              
            pos[i] = pos[i] * (b-a) + a 

        return pos


    def _create_node_text_info(self,
                            inv_prods: list[InventoryProduct],
                            node_pos: list[float, float],
                            inv_node_id=None) -> list:
        """Create info for node."""
        text_els = []
        # Offset to account for the ID line and separator line
        inv_offset = 0
        if inv_node_id is not None:
            inv_offset = 2
            
            # Create text for ID
            id_text = self.font.render(f"ID:{COLUMN_SEP}{inv_node_id}", True, WHITE)
            id_text_pos = (
                node_pos[0] - CIRCLE_RADIUS, 
                node_pos[1] + NODE_INFO_OFFSET)
            text_els.append([id_text, id_text_pos])

            # Create text for separating ID from inventory
            line_text = self.font.render(f"----------", True, WHITE)
            line_text_pos = (
                node_pos[0] - CIRCLE_RADIUS,
                node_pos[1] + NODE_INFO_OFFSET + self.args.font_size)
            text_els.append([line_text, line_text_pos])

        # Create text element for every InventoryProduct
        for i, inv_prod in enumerate(inv_prods):
            inv_prod_str = f"{inv_prod.sku_id}:{COLUMN_SEP}{inv_prod.quantity}"
            inv_prod_text = self.font.render(inv_prod_str, True, WHITE)
            inv_prod_pos = (
                node_pos[0] - CIRCLE_RADIUS,
                node_pos[1] + NODE_INFO_OFFSET + self.args.font_size * (i+inv_offset)) 
            text_els.append([inv_prod_text, inv_prod_pos])

        return text_els

    def _render_text(self, info_text):
        for text_info_line in info_text:
            self.screen.blit(*text_info_line)

    def render_order(self,
                    demand_node: DemandNode,
                    policy_results: PolicyResults = None):
        # Check if should quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        
        # Set background black
        self.screen.fill(BLACK)

        # Render the inventory nodes
        for inv_node in self._inv_nodes:

            inv_node_pos = self._sim_to_screen(
                [inv_node.loc.coords.x, inv_node.loc.coords.y])
            pygame.draw.circle(
                self.screen, 
                BLUE, 
                inv_node_pos, 
                CIRCLE_RADIUS)

            # Render the inventory node info
            inv_info_text = self._create_node_text_info(
                inv_node.inv.items(),
                inv_node_pos)
            self._render_text(inv_info_text)

        # Render the demand node
        demand_node_pos = self._sim_to_screen(
            [demand_node.loc.coords.x, demand_node.loc.coords.y])
        pygame.draw.circle(
            self.screen,
            GREEN,
            demand_node_pos, 
            CIRCLE_RADIUS)

        # Render the demand node order info
        demand_info_text = self._create_node_text_info(
            demand_node.inv.items(),
            demand_node_pos)
        self._render_text(demand_info_text)

        # Render the fulfillment
        if policy_results:
            for fulfillment in policy_results.fulfill_plan.fulfillments():
                inv_node_pos = self._sim_to_screen([
                    self._inv_nodes[fulfillment.inv_node_id].loc.coords.x,
                    self._inv_nodes[fulfillment.inv_node_id].loc.coords.y])

                fulfill_info_pos = [
                        (inv_node_pos[0] + demand_node_pos[0]) // 2,
                        (inv_node_pos[1] + demand_node_pos[1])//2]

                fulfill_info_text = self._create_node_text_info(
                    fulfillment.inv.items(),
                    fulfill_info_pos)

                pygame.draw.line(
                    self.screen,
                    RED,
                    inv_node_pos,
                    demand_node_pos,
                    LINE_WIDTH)

                self._render_text(fulfill_info_text)
        
        # Render reward
        self._total_reward += sum([exp.reward for exp in policy_results.exps])
        total_reward_text = self.font.render(
            "Total reward: " + str(self._total_reward),
            True,
            WHITE)
        self.screen.blit(total_reward_text, (0, 0))
        pygame.time.delay(1000)
        pygame.display.flip()