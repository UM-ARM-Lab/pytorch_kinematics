import torch
import pytorch_kinematics as pk


def test_fkik():
    data = '<robot name="test_robot">' \
           '<link name="link1" />' \
           '<link name="link2" />' \
           '<link name="link3" />' \
           '<joint name="joint1" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link1"/>' \
           '<child link="link2"/>' \
           '</joint>' \
           '<joint name="joint2" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link2"/>' \
           '<child link="link3"/>' \
           '</joint>' \
           '</robot>'
    chain = pk.build_serial_chain_from_urdf(data, 'link3')
    th1 = torch.tensor([0.42553542, 0.17529176])
    tg = chain.forward_kinematics(th1)
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    assert torch.allclose(pos, torch.tensor([[1.91081784, 0.41280851, 0.0000]]))
    assert torch.allclose(rot, torch.tensor([[0.95521418, 0.0000, 0.0000,  0.2959153]]))
    print(tg)
    # TODO test batch kinematics
    # TODO implement and test inverse kinematics
    # th2 = chain.inverse_kinematics(tg)
    # self.assertTrue(np.allclose(th1, th2, atol=1.0e-6))


if __name__ == "__main__":
    test_fkik()
