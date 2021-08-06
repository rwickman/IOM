import pygame, sys

from nodes import InventoryProduct, InventoryNode, DemandNode
from policy import PolicyResults 

pygame.init()

COLUMN_SEP = "    "
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255,0,0)

INV_NODE_COLOR = (134, 197, 218)
INV_NODE_OUTLINE_COLOR = (80, 165, 193)


DEMAND_NODE_COLOR = (134, 218, 155)#(239,190,125)
DEMAND_NODE_OUTLINE_COLOR = (80, 193, 108)

# The y offset below at which node info will display
NODE_INFO_OFFSET = 32

# Radius of the inventory node circles
INV_NODE_RADIUS = 35
INV_NODE_OUTLINE_RADIUS = 3

MIN_INV_NODE_RADIUS = 15

# Determines if you should render an inventory node when stock is exhausted
RENDER_NODE_WITHOUT_STOCK = False

# Radius of the demand node circles
DEMAND_NODE_RADIUS = 20
DEMAND_NODE_OUTLINE_RADIUS = 3


LINE_WIDTH = 5
TOTAL_COST_ROUND_DIGITS = 2
INV_NODE_ID_OFFSET = (-6, -10)

DEMAND_GROW_STEPS = 5

# Amount of frames to interpolate a product from an inventory node to a demand node
FULFILL_STEPS = 10

# Amount of frames to delay before rendering another item in an order
FULFILL_DELAY_STEPS = 2
FULFILL_ICON_OFFSET = (-11, -11)

FULFILL_NODE_COLOR = (237, 0, 0)#(218, 161, 134)#(255,0,0)
FULFILL_NODE_OUTLINE_COLOR = (212, 42, 42)

FULFILL_NODE_RADIUS = 17
FULFILL_NODE_OUTLINE_RADIUS = 3

# Total amount of frames in an order
NUM_ORDER_FRAMES = 20

# Amount of time to wait between orders
DELAY_MS = 0

# Delay between frames when growing node
GROW_FRAME_DELAY = 10

# Delay between frames when fulfilling order
FULFILL_FRAME_DELAY = 3

# # Percentage of screen size the icon should cover
# ICON_SCALE = 0.01

class Visual:
    def __init__(self, args, inv_nodes: list[InventoryNode]):
        self.args = args
        self._inv_nodes = inv_nodes

        # Get max inv size
        self.max_inv_size = max(inv_node.inv.inv_size for inv_node in self._inv_nodes)

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

        # Create the background
        self.bg_img = pygame.image.load("../pics/maps/pacific_northwest_blank.png")
        # Resize background
        self.bg_img = pygame.transform.scale(self.bg_img, (self.width, self.height))

        # Load the icons
        self.icons = {
            "shirt" : pygame.image.load("../pics/icons/shirt_icon.png"),
            "shoe" : pygame.image.load("../pics/icons/shoe_icon.png")
        }

        # Resize the icons
        for key, val in self.icons.items():
            print(key, val)
            self.icons[key] = pygame.transform.scale(
                val, (self.args.font_size, self.args.font_size))

        # Create the screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.font = pygame.font.Font('freesansbold.ttf', self.args.font_size)
        self.font_title = pygame.font.Font('freesansbold.ttf', self.args.font_size * 2)
        self._total_reward = 0
        self._timestep = 0
    
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


    def _render_icon(self, sku_id: int, pos: tuple[float, float]):
            """Render an icon."""
            if sku_id == 0:
                self.screen.blit(self.icons["shirt"], pos)
            elif sku_id == 1:
                self.screen.blit(self.icons["shoe"], pos)
            else:
                self.screen.blit(self.font.render(str(sku_id), True, BLACK), pos)

        
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
            id_text = self.font.render(f"ID:{COLUMN_SEP}{inv_node_id}", True, BLACK)
            id_text_pos = (
                node_pos[0] - INV_NODE_RADIUS, 
                node_pos[1] + NODE_INFO_OFFSET)
            text_els.append([id_text, id_text_pos])

        # Create text element for every InventoryProduct
        for i, inv_prod in enumerate(inv_prods):
            inv_prod_str = f"{COLUMN_SEP}:{COLUMN_SEP}{inv_prod.quantity}"
            inv_prod_text = self.font.render(inv_prod_str, True, BLACK)
            inv_prod_pos = (
                node_pos[0] - INV_NODE_RADIUS,
                node_pos[1] + NODE_INFO_OFFSET + self.args.font_size * (i+inv_offset)) 
            text_els.append([inv_prod_text, inv_prod_pos])

            # Render the product icon
            self._render_icon(inv_prod.sku_id, inv_prod_pos)

        return text_els

    def _render_text(self, info_text):
        for text_info_line in info_text:
            self.screen.blit(*text_info_line)

    def reset(self):
        self._total_reward = 0
        self._timestep = 0

    
    def _render_demand_node(self, demand_node: DemandNode, cur_step=0):
        # Render the demand node
        demand_node_pos = self._sim_to_screen(
            [demand_node.loc.coords.x, demand_node.loc.coords.y])

        
        if cur_step - 1 >= DEMAND_GROW_STEPS:
            # Render the demand node order info
            demand_info_text = self._create_node_text_info(
                demand_node.inv.items(),
                demand_node_pos)
            
            self._render_text(demand_info_text)

        # Scale factor that allows it to grow
        node_scale_factor = min((cur_step+1)/DEMAND_GROW_STEPS, 1)
        
        # Render the outline circle
        pygame.draw.circle(
            self.screen,
            DEMAND_NODE_OUTLINE_COLOR,
            demand_node_pos, 
            (DEMAND_NODE_RADIUS + DEMAND_NODE_OUTLINE_RADIUS) * node_scale_factor)
        
        # Render inner circle
        pygame.draw.circle(
            self.screen,
            DEMAND_NODE_COLOR,
            demand_node_pos, 
            DEMAND_NODE_RADIUS * node_scale_factor)

    def _render_inv_nodes(self):
        # Render the inventory nodes
        for inv_node in self._inv_nodes:
            if inv_node.inv.inv_size == 0 and not RENDER_NODE_WITHOUT_STOCK:
                continue

            # Render the circle for the inventory node
            inv_node_pos = self._sim_to_screen(
                [inv_node.loc.coords.x, inv_node.loc.coords.y])
            
            # Scale the size based on the amount of inventory left
            inv_node_radius = inv_node.inv.inv_size / self.max_inv_size 
            inv_node_radius = inv_node_radius * (INV_NODE_RADIUS - MIN_INV_NODE_RADIUS) + MIN_INV_NODE_RADIUS

            # Render the outline circle
            pygame.draw.circle(
                self.screen, 
                INV_NODE_OUTLINE_COLOR, 
                inv_node_pos, 
                inv_node_radius + INV_NODE_OUTLINE_RADIUS)
            
            # Render the inner circle
            pygame.draw.circle(
                self.screen, 
                INV_NODE_COLOR, 
                inv_node_pos, 
                inv_node_radius)
            
            
            node_id_text = self.font.render(
                str(inv_node.inv_node_id), True, BLACK)
            self.screen.blit(
                node_id_text,
                (inv_node_pos[0] + INV_NODE_ID_OFFSET[0], inv_node_pos[1] + INV_NODE_ID_OFFSET[1]))

            # Render the inventory node info
            inv_info_text = self._create_node_text_info(
                inv_node.inv.items(),
                inv_node_pos)
            self._render_text(inv_info_text)

    def _render_fulfillment(self,
                            demand_node: DemandNode,
                            policy_results: PolicyResults,
                            cur_step: int):
        """Render the products in a fulfillment."""
        # Render the demand node
        demand_node_pos = self._sim_to_screen(
            [demand_node.loc.coords.x, demand_node.loc.coords.y])

        # Amount of frames since fulfillment animation has started
        cur_fulfill_step = cur_step - DEMAND_GROW_STEPS

        for fulfillment in policy_results.fulfill_plan.fulfillments():
            inv_node_pos = self._sim_to_screen([
                self._inv_nodes[fulfillment.inv_node_id].loc.coords.x,
                self._inv_nodes[fulfillment.inv_node_id].loc.coords.y])

            cur_prod_idx = 0
            for inv_prod in fulfillment.inv.items():
                for q_j in range(inv_prod.quantity):
                    start_step = cur_prod_idx * FULFILL_DELAY_STEPS
                    end_step = start_step + FULFILL_STEPS
                    
                    prod_inter_val = (cur_fulfill_step - start_step) / (end_step - start_step)
                    if prod_inter_val >= 0 and prod_inter_val <= 1:
                        prod_pos = [
                            (1-prod_inter_val) * inv_node_pos[0] + prod_inter_val * demand_node_pos[0],
                            (1-prod_inter_val) * inv_node_pos[1] + prod_inter_val * demand_node_pos[1]]
                        

                        # Render the outer circle
                        pygame.draw.circle(
                            self.screen,
                            FULFILL_NODE_OUTLINE_COLOR,
                            prod_pos, 
                            FULFILL_NODE_RADIUS + FULFILL_NODE_OUTLINE_RADIUS)
                        
                        # Render the inner circle
                        pygame.draw.circle(
                            self.screen,
                            FULFILL_NODE_COLOR,
                            prod_pos, 
                            FULFILL_NODE_RADIUS)

                        # Create the icon
                        icon_pos = [
                            prod_pos[0] + FULFILL_ICON_OFFSET[0],    
                            prod_pos[1] + FULFILL_ICON_OFFSET[1]
                        ]
                        
                        self._render_icon(inv_prod.sku_id, icon_pos)
                    
                    cur_prod_idx += 1


    def render_order(self,
                    demand_node: DemandNode,
                    policy_results: PolicyResults = None,
                    policy_name: str = None):

        # Update the order timestep
        self._timestep += 1
        time_str = str(self._timestep)

        # Update the total reward
        self._total_reward += sum([exp.reward for exp in policy_results.exps])
        
        # Get max fulfillment size
        max_fulfill = max(f.inv.inv_size for f in policy_results.fulfill_plan.fulfillments())
        
        if policy_results:
            num_order_frames = DEMAND_GROW_STEPS + max_fulfill * FULFILL_DELAY_STEPS + FULFILL_STEPS
        else:
            num_order_frames = DEMAND_GROW_STEPS

        # Create order frames
        for cur_step in range(num_order_frames):
                
            # Check if should quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
            
            # Set the background
            self.screen.fill(BLACK)
            self.screen.blit(self.bg_img, (0,0))
            
            self._render_inv_nodes()

            self._render_demand_node(demand_node, cur_step)

            # Render the fulfillment, once demand node is full size
            if cur_step >= DEMAND_GROW_STEPS and policy_results:
                self._render_fulfillment(demand_node, policy_results, cur_step)

            
            # Render the policy name
            policy_name_text = self.font_title.render(policy_name.upper(), True, BLACK)
            self.screen.blit(policy_name_text, (self.width//2 - 50, 50))

            # Render reward
            total_reward_text = self.font_title.render(
                "Total Cost: $" + str(round(self._total_reward, TOTAL_COST_ROUND_DIGITS) * -1),
                True,
                BLACK)
            self.screen.blit(total_reward_text, (0, 0))
            
            # Rendoer the order timestep
            time_text = self.font_title.render(
                time_str,
                True,
                BLACK)
            self.screen.blit(time_text, (self.width - self.args.font_size*3, 0))

            # Create new display
            pygame.display.flip()

            if cur_step < DEMAND_GROW_STEPS:
                pygame.time.delay(GROW_FRAME_DELAY)
            else:
                pygame.time.delay(FULFILL_FRAME_DELAY)

            
        if DELAY_MS >= 0:
            pygame.time.delay(DELAY_MS)