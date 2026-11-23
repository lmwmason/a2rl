from concurrent.futures import ThreadPoolExecutor

from tools.elodin_heat import HeatTimeline, intersects


def test_crossing_aircraft_make_contact_between_samples():
    assert intersects((-1, 0, 1), (1, 0, 1), (1, 0, 1), (-1, 0, 1))
    assert not intersects((-1, 0, 1), (1, 0, 1.6), (1, 0, 1), (-1, 0, 1.6))


def test_peers_and_contacts_follow_shared_sample_numbers():
    heat = HeatTimeline()
    positions = [(-1, 0, 1), (1, 0, 1), (0, 0, 3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: heat.update(item[0], 0, item[1]), enumerate(positions)))
    assert results[0] == [positions[1:], False]

    assert heat.update(0, 1, (1, 0, 1)) == [positions[1:], False]
    heat.update(1, 1, (-1, 0, 1))
    _, third_contact = heat.update(2, 1, positions[2])
    _, first_contact = heat.update(0, 2, (2, 0, 1))
    assert first_contact and not third_contact
