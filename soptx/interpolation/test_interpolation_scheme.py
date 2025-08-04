from fealpy.decorator import variantmethod

class InterpolationSchemeTest():
    def __init__(self) -> None:
        pass

    @variantmethod('test')
    def run(self, 
            interpolation_order: int, 
        ) -> None:

        # 参数设置
        nx, ny = 1, 1
        
        # 设置 pde 和网格
        from soptx.model.mbb_beam_2d import HalfMBBBeam2dData
        pde = HalfMBBBeam2dData(
                            domain=[0, nx, 0, ny],
                            T=-1.0, E=1.0, nu=0.3,
                            enable_logging=False
                        )
        pde.init_mesh.set('uniform_quad')

        
        opt_mesh = pde.init_mesh(nx=nx, ny=ny)

        from soptx.interpolation.interpolation_scheme import MaterialInterpolationScheme
        interpolation_scheme = MaterialInterpolationScheme(
                                    density_location='interpolation_point',
                                    interpolation_method='msimp',
                                    options={
                                        'penalty_factor': 3.0,
                                        'void_youngs_modulus': 1e-9,
                                        'target_variables': ['E']
                                    },
                                )

        rho_nodes = interpolation_scheme.setup_density_distribution(
                                                mesh=opt_mesh,
                                                relative_density=0,
                                                interpolation_order=interpolation_order,
                                            )
        
        rho_nodes[1] = 1.0  # 左上角节点
        rho_nodes[2] = 1.0  # 右下角节点
        density_space = rho_nodes.space
        
if __name__ == "__main__":
    test = InterpolationSchemeTest()
    test.run.set('test')
    test.run(interpolation_order=2)